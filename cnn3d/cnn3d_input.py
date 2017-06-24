#!/usr/bin/env python

import numpy as np 
import tensorflow as tf
from scipy.ndimage.interpolation import map_coordinates
import pandas as pd
import random
from math import pi, sin, cos

import time
from glob import glob
import os


ROTATION_RANGE = [0, 360]
STRENTCH_RANGE = [0.8, 1.2]
POS_SHIFT_RANGE = [0., 0.1]  # relative shift range for positive samples
NEG_SHIFT_RANGE = [0, 2]  # absolute shift range for negative samples
NOISE_RANGE = [0.8, 1.2] 

class CNN3dTrainInput(object):
    """docstring for LUNATrainInput"""
    def __init__(self, 
                 data_dir,
                 annotation_file, 
                 prediction_file, 
                 min_nodule=10, 
                 max_nodule=100,
                 macro_batch_size=10,
                 macro_batch_reuse=100,
                 micro_batch_size=100,
                 sample_ratio=[0.5,0.5],
                 sample_size=48,  # image sample size in x,y,z dimention
                 debug=False,
                 verbose=False):
        self.data_dir = data_dir
        self.annotation_file = annotation_file  # manually annotation csv file 
        self.prediction_file = prediction_file  # predicted nodule csv file
        self.min_nodule = min_nodule  # in mm
        self.max_nodule = max_nodule  # in mm
        self.macro_batch_size = macro_batch_size
        self.macro_batch_reuse = macro_batch_reuse
        self.micro_batch_size = micro_batch_size
        self.sample_ratio = sample_ratio  # sampling ratio of nodule, tissue, border
        self.sample_size = sample_size
        self.debug = debug
        self.verbose = verbose

        # get positive samples from annotation file
        ref_df = pd.read_csv(self.annotation_file)
        diameters = ref_df['diameter_mm'].values
        pos_idx = np.where((diameters > self.min_nodule) * \
            (diameters < self.max_nodule))[0]
        self.pos_rows = ref_df.iloc[pos_idx]
        if self.debug:
            print('positive samples: \n', self.pos_rows)

        # get negative samples from prediction file
        pred_df = pd.read_csv(self.prediction_file)
        diameters = pred_df['diameter_mm'].values
        neg_idx = np.where(diameters == 0.)[0]
        self.neg_rows = pred_df.iloc[neg_idx]
        if self.debug:
            print('negative samples: \n', self.neg_rows)

        # collect seriesuids
        self.seriesuids = []
        self.seriesuids += list(set(self.pos_rows['seriesuid'].values.tolist()))
        # self.seriesuids += list(set(self.neg_rows['seriesuid'].values.tolist()))

        # init macro and micro batch
        self.macro_batch = None
        self.macro_batch_count = 0
        self.micro_batch = None

    def get_sample(self, image, x, y, z, 
        sample_type=None, radius_px=0.):
        sample = None
        s_z, s_y, s_x = image.shape
        ss_x, ss_y, ss_z = [self.sample_size] * 3

        # data augmentation
        angle = float(random.randint(ROTATION_RANGE[0], 
            ROTATION_RANGE[1])) / 180. * pi
        strentch_ratio_x = random.uniform(STRENTCH_RANGE[0], 
            STRENTCH_RANGE[1])
        strentch_ratio_y = random.uniform(STRENTCH_RANGE[0], 
            STRENTCH_RANGE[1])
        if sample_type == 'P': # positive sample
            shift_x = random.uniform(POS_SHIFT_RANGE[0]*radius_px, 
                POS_SHIFT_RANGE[1]*radius_px)
            shift_y = random.uniform(POS_SHIFT_RANGE[0]*radius_px, 
                POS_SHIFT_RANGE[1]*radius_px)
        elif sample_type == 'N':  # negative sample
            shift_x = random.uniform(NEG_SHIFT_RANGE[0], 
                NEG_SHIFT_RANGE[1])
            shift_y = random.uniform(NEG_SHIFT_RANGE[0], 
                NEG_SHIFT_RANGE[1])
        flip_rv = random.randint(0, 4)

        try:
            for offset in range(int(-ss_z//2), int(-ss_z//2)+ss_z):
                slice_y, slice_x = np.indices(image.shape[1:])
                crop_y = slice_y[int(y-ss_y//2):int(y-ss_y//2)+ss_y,
                                 int(x-ss_x//2):int(x-ss_x//2)+ss_x]
                crop_x = slice_x[int(y-ss_y//2):int(y-ss_y//2)+ss_y,
                                 int(x-ss_x//2):int(x-ss_x//2)+ss_x]
                _y = (crop_y - y).reshape(-1)
                _x = (crop_x - x).reshape(-1)
                _xy = np.vstack((_x, _y))
                RM = np.array([[cos(angle), -sin(angle)],
                               [sin(angle), cos(angle)]])
                _rxy = RM.dot(_xy) 
                crop_x = np.reshape(_rxy[0], crop_x.shape) + x
                crop_y = np.reshape(_rxy[1], crop_y.shape) + y
                crop_x = (crop_x - x) / strentch_ratio_x + x
                crop_y = (crop_y - y) / strentch_ratio_y + y 
                crop_x += shift_x
                crop_y += shift_y           

                sample_slice = image[int(z+offset),:,:]
                crop_slice = map_coordinates(sample_slice, [crop_y, crop_x], order=0)
                if flip_rv == 2:
                    crop_slice = np.flipud(crop_slice)
                elif flip_rv == 3:
                    crop_slice == np.fliplr(crop_slice)
                if sample is None:
                    sample = crop_slice[np.newaxis,...]
                else:
                    sample = np.vstack((sample, crop_slice[np.newaxis,...]))
        except:
            if self.debug or self.verbose:
                print('catch exception during sample generation')
            return None
        # add some noise
        noise = np.random.uniform(low=NOISE_RANGE[0], 
            high=NOISE_RANGE[1], size=sample.shape)
        sample = (sample.astype(np.float32) * noise.astype(np.float32)).astype(np.int16)
        return sample

    def next_micro_batch(self):
        if self.debug:
            start = time.time()
        if self.macro_batch is None or \
            self.macro_batch_count == self.macro_batch_reuse:
            self.next_macro_batch()
            self.macro_batch_count = 0

        count = 0
        # init samples as dummy samples
        samples = np.ones((self.micro_batch_size,
            self.sample_size, self.sample_size, self.sample_size)) * -1000
        labels = np.zeros(self.micro_batch_size, dtype=np.int16)

        nb_pos_samples = int(self.micro_batch_size * self.sample_ratio[0])
        nb_neg_samples = int(self.micro_batch_size * self.sample_ratio[1])

        # positive nodule samples
        for i in range(nb_pos_samples):
            # randomly pick a CT scan
            idx = random.randint(0, len(self.macro_batch)-1)
            seriesuid, data = self.macro_batch[idx]
            spacing = data['spacing']  # in x, y, z order
            origin = data['origin']  # in x, y, z order
            image = data['image']  # in z, y, x order

            # randomly pick a nodule in this CT record
            rows = self.pos_rows[self.pos_rows['seriesuid'] == seriesuid]
            if len(rows) == 0:
                if self.debug or self.verbose:
                    print('skipping %s for positive sample generation' %
                        seriesuid)
                continue
            idx = random.randint(0, len(rows)-1)
            row = rows.iloc[idx]
            if self.debug or self.verbose:
                print('generating positive sample from %s' % 
                    str(row))
            coord_mm = np.asarray([row['coordX'], row['coordY'], row['coordZ']])
            diameter = row['diameter_mm']
            r = int(diameter / spacing[0] / 2)  # radius in pixel
            coord_pixel = (coord_mm - origin) / spacing
            x, y, z = coord_pixel
            sample = self.get_sample(image, x, y, z,
                sample_type='P', radius_px=r)
            if sample is None:
                continue
            samples[count,:,:,:] = sample
            labels[count] = 1  # 1 for nodule, 0 for else
            count += 1

            if self.debug:
                print('saving %d sample' % count)
                np.save('positive_sample_%d.npy' % count, sample)

        # negative samples
        for i in range(nb_neg_samples):
            # randomly pick a CT scan
            idx = random.randint(0, len(self.macro_batch)-1)
            seriesuid, data = self.macro_batch[idx]
            spacing = data['spacing']  # in x, y, z order
            origin = data['origin']  # in x, y, z order
            image = data['image']  # in z, y, x order

            # randomly pick a false positive in this CT scan
            rows = self.neg_rows[self.neg_rows['seriesuid'] == seriesuid]
            if len(rows) == 0:
                if self.debug or self.verbose:
                    print('skipping %s for negative sample generation' %
                        seriesuid)
                continue
            idx = random.randint(0, len(rows)-1)
            row = rows.iloc[idx]
            if self.debug or self.verbose:
                print('generating negative sample from %s' % 
                    str(row))
            coord_mm = np.asarray([row['coordX'], row['coordY'], row['coordZ']])
            coord_pixel = (coord_mm - origin) / spacing
            x, y, z = coord_pixel
            sample = self.get_sample(image, x, y, z,
                sample_type='N')
            if sample is None:
                continue
            samples[count,:,:,:] = sample
            labels[count] = 0  # 1 for nodule, 0 for else
            count += 1

            if self.debug:
                print('saving %d sample' % count)
                np.save('negative_sample_%d.npy' % count, sample)

        self.macro_batch_count += 1
        if self.debug:
            end = time.time()
            time_cost = end - start
            print('time elapsed loading micro batch: %.2f sec' % time_cost)
        samples = np.asarray(samples, dtype=np.float32)
        samples = samples.reshape((self.micro_batch_size, 
            self.sample_size, self.sample_size, self.sample_size, 1))
        labels = np.asarray(labels, dtype=np.int8)
        self.micro_batch = [samples, labels]
        return self.micro_batch

    def next_macro_batch(self):
        if self.debug or self.verbose:
            start = time.time()
        macro_batch_size = min(self.macro_batch_size, len(self.seriesuids))
        random_idx = random.sample(range(len(self.seriesuids)), macro_batch_size)
        self.macro_batch = []
        for i in random_idx:
            seriesuid = self.seriesuids[i]
            fname = '%s/%s.npz' % (self.data_dir, seriesuid)
            if self.debug or self.verbose:
                print 'loading %s' % fname
            data = np.load(fname)
            spacing = data['spacing']
            origin = data['origin']
            image = data['data']
            mask = data['mask']
            data = {'spacing': spacing,
                    'origin': origin,
                    'image': image,
                    'mask': mask}
            self.macro_batch.append((seriesuid, data))
        if self.debug or self.verbose:
            end = time.time()
            time_cost = end - start
            print('time elapsed loading macro batch: %.2f sec' % time_cost)


if __name__ == '__main__':
    data_dir = '/Volumes/SPIDATA/TIANCHI/train_processed_border-5_gradient-search'
    annotation_file = '/Volumes/SPIDATA/TIANCHI/csv/train/annotations.csv'
    prediction_file = '/Users/lixuanxuan/Desktop/TIANCHI/train_prediction/0608/train-0608-min-dist_10.csv'
    show = True
    train_input = CNN3dTrainInput(data_dir, 
                                annotation_file,
                                prediction_file,
                                macro_batch_size=3,
                                min_nodule=30, 
                                max_nodule=100, 
                                sample_ratio=[0.5,0.5],
                                debug=False,
                                verbose=True)
    data, labels = train_input.next_micro_batch()
    if show:
        import matplotlib.pyplot as plt

        rv = random.randint(0, len(data))
        plt.imshow(data[rv][0,:,:], cmap='gray')
        plt.title(labels[rv])
        plt.clim((-600, 200))
        plt.show()
