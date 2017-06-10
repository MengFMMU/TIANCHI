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


class LUNATrainInput(object):
    """docstring for LUNATrainInput"""
    def __init__(self, 
                 data_dir, 
                 csv_file, 
                 min_nodule=10, 
                 max_nodule=100,
                 macro_batch_size=10,
                 macro_batch_reuse=100,
                 micro_batch_size=100,
                 sample_ratio=[0.5,0.3,0.2,0.0,0.0],
                 sample_size_xy=48,  # image sample size in x,y dimention
                 sample_size_hz=1,  # half image sample size in z, actual size = 2 * hz + 1
                 random_rotation=True,
                 rotation_range=[0, 360],
                 random_strentch=True,
                 strentch_range=[0.8, 1.2],
                 random_shift=True,
                 shift_range=[0, 2],
                 random_flip=True,
                 random_noise=True,
                 noise_ratio=[0.8, 1.2],
                 exclude_tol=1.2,  # exclude volume when generating negative samples
                 negative_samples_from=3,  # generating negative samples from N CT scans
                 false_sample_dir=None,
                 false_sample_date=None,
                 feed_false_samples=False,
                 debug=False,
                 verbose=False):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.min_nodule = min_nodule  # in mm
        self.max_nodule = max_nodule  # in mm
        self.macro_batch_size = macro_batch_size
        self.macro_batch_reuse = macro_batch_reuse
        self.micro_batch_size = micro_batch_size
        self.sample_ratio = sample_ratio  # sampling ratio of nodule, tissue, border
        self.sample_size_xy = sample_size_xy
        self.sample_size_hz = sample_size_hz
        self.random_rotation = random_rotation
        self.rotation_range = rotation_range
        self.random_strentch = random_strentch
        self.strentch_range = strentch_range
        self.random_shift = random_shift
        self.shift_range = shift_range
        self.random_flip = random_flip
        self.random_noise = random_noise
        self.noise_ratio = noise_ratio
        self.exclude_tol = exclude_tol
        self.negative_samples_from = negative_samples_from
        self.false_sample_dir = false_sample_dir
        self.false_sample_date = false_sample_date
        self.feed_false_samples = feed_false_samples
        self.debug = debug
        self.verbose = verbose

        # load and parse annotation csv
        df = pd.read_csv(self.csv_file)
        diameters = df['diameter_mm'].values
        valid_idx = np.where((diameters > self.min_nodule) * \
            (diameters < self.max_nodule))[0]
        valid_rows = df.iloc[valid_idx]
        valid_seriesuid = valid_rows['seriesuid'].values
        # remove duplicates
        self.seriesuids = list(set(valid_seriesuid.tolist()))
        self.rows = valid_rows

        # init macro and micro batch
        self.macro_batch = None
        self.macro_batch_count = 0
        self.micro_batch = None

    def get_sample(self, image, x, y, z):
        sample = None
        sz, sy, sx = image.shape
        size_xy = self.sample_size_xy  # size in x, y dimenstion
        size_hz = self.sample_size_hz  # half size in z dimention
        if (z-size_hz) < 0 or (z+size_hz) > sz-1:
            return None
        if (x-size_xy/2) < 0 or (x+size_xy/2) > sx-1:
            return None
        if (y-size_xy/2) < 0 or (y+size_xy/2) > sy-1:
            return None

        # data augmentation
        if self.random_rotation:
            min_angle = self.rotation_range[0]
            max_angle = self.rotation_range[1]
            angle = float(random.randint(min_angle, max_angle)) / 180. * pi
        if self.random_strentch:
            min_ratio = self.strentch_range[0]
            max_ratio = self.strentch_range[1]
            strentch_ratio_x = random.uniform(min_ratio, max_ratio)
            strentch_ratio_y = random.uniform(min_ratio, max_ratio)
        if self.random_shift:
            min_shift = self.shift_range[0]
            max_shift = self.shift_range[1]
            shift_x = random.uniform(min_shift, max_shift)
            shift_y = random.uniform(min_shift, max_shift)
        if self.random_flip:
            flip_rv = random.randint(0, 4)

        for offset in range(-self.sample_size_hz, self.sample_size_hz+1):
            slice_y, slice_x = np.indices(image.shape[1:])
            crop_y = slice_y[y-self.sample_size_xy//2:y-self.sample_size_xy//2+self.sample_size_xy,
                             x-self.sample_size_xy//2:x-self.sample_size_xy//2+self.sample_size_xy]
            crop_x = slice_x[y-self.sample_size_xy//2:y-self.sample_size_xy//2+self.sample_size_xy,
                             x-self.sample_size_xy//2:x-self.sample_size_xy//2+self.sample_size_xy]
            # data augmentation
            if self.random_rotation:
                _y = (crop_y - y).reshape(-1)
                _x = (crop_x - x).reshape(-1)
                _xy = np.vstack((_x, _y))
                RM = np.array([[cos(angle), -sin(angle)],
                               [sin(angle), cos(angle)]])
                _rxy = RM.dot(_xy)  # rotate coordinates
                crop_x = np.reshape(_rxy[0], crop_x.shape) + x
                crop_y = np.reshape(_rxy[1], crop_y.shape) + y
            if self.random_strentch:
                crop_x = (crop_x - x) / strentch_ratio_x + x
                crop_y = (crop_y - y) / strentch_ratio_y + y 
            if self.random_shift:
                crop_x += shift_x
                crop_y += shift_y

            sample_slice = image[z+offset,:,:]
            crop_slice = map_coordinates(sample_slice, [crop_y, crop_x], order=0)
            if self.random_flip:
                if flip_rv == 2:
                    crop_slice = np.flipud(crop_slice)
                elif flip_rv == 3:
                    crop_slice == np.fliplr(crop_slice)
            if sample is None:
                sample = crop_slice[np.newaxis,...]
            else:
                sample = np.vstack((sample, crop_slice[np.newaxis,...]))
        if self.random_noise:
            noise = np.random.uniform(low=self.noise_ratio[0], 
                high=self.noise_ratio[1], size=sample.shape)
            sample = (sample.astype(np.float32) * noise.astype(np.float32)).astype(np.int16)
        return sample

    def next_micro_batch(self):
        if self.debug:
            start = time.time()
        if self.macro_batch is None or \
            self.macro_batch_count == self.macro_batch_reuse:
            self.next_macro_batch()
            self.macro_batch_count = 0

        samples = []
        labels = []

        nb_nodule_samples = int(self.micro_batch_size * self.sample_ratio[0])
        nb_tissue_samples = int(self.micro_batch_size * self.sample_ratio[1])
        nb_border_samples = int(self.micro_batch_size * self.sample_ratio[2])
        nb_false_sample_nodules = int(self.micro_batch_size * self.sample_ratio[3])  # false sample nodules, real nodule but classified as nonnodule
        nb_false_sample_nonnodules = int(self.micro_batch_size * self.sample_ratio[4])  # false sample nonnodules, nonnodule but classified as nodule
        # positive nodule samples
        for i in range(nb_nodule_samples):
            # randomly pick a CT record
            idx = random.randint(0, len(self.macro_batch)-1)
            seriesuid, data = self.macro_batch[idx]
            spacing = data['spacing']  # in x, y, z order
            origin = data['origin']  # in x, y, z order
            image = data['image']  # in z, y, x order
            sz, sy, sx = image.shape

            # randomly pick a nodule in this CT record
            rows = self.rows[self.rows['seriesuid'] == seriesuid]
            idx = random.randint(0, len(rows)-1)
            row = rows.iloc[idx]
            assert row['seriesuid'] == seriesuid
            assert row['diameter_mm'] > self.min_nodule
            assert row['diameter_mm'] < self.max_nodule

            coord_mm = np.asarray([row['coordX'], row['coordY'], row['coordZ']])
            diameter = row['diameter_mm']
            r = int(diameter / spacing[0] / 2)  # radius in pixel
            coord_pixel = (coord_mm - origin) / spacing
            x, y, z = coord_pixel
            if min(x-r, y-r, z-r) < 0 or min(sx-1-x-r, sy-1-y-r, sz-1-z-r) < 0:
                if self.verbose:
                    print('WARNING!!! Wrong nodule coordinate: %s' % (str(row)))
                continue
            sample = self.get_sample(image, x, y, z)
            if sample is None:
                continue
            samples.append(sample)
            labels.append(1)  # 1 for nodule, 0 for else

            if self.debug:
                print('saving %d sample' % count)
                np.save('nodule_sample_%d.npy' % count, sample)

        # negative tissue samples
        nb_samples_per_scan = int(nb_tissue_samples / self.negative_samples_from)
        last_samples = nb_tissue_samples - (self.negative_samples_from-1) * nb_samples_per_scan
        for i in range(self.negative_samples_from):
            # randomly pick a CT record
            idx = random.randint(0, len(self.macro_batch)-1)
            seriesuid, data = self.macro_batch[idx]
            spacing = data['spacing']  # in x, y, z order
            origin = data['origin']  # in x, y, z order
            image = data['image']  # in z, y, x order
            mask = data['mask']  # in z, y, x order
            sz, sy, sx = image.shape

            # exclude nodule volume 
            rows = self.rows[self.rows['seriesuid'] == seriesuid]
            for j in range(len(rows)):
                row = rows.iloc[j]
                coord_mm = np.asarray([row['coordX'], row['coordY'], row['coordZ']])
                diameter_mm = row['diameter_mm']
                coord_pixel = (coord_mm - origin) / spacing
                x, y, z = coord_pixel
                diameter_pixel = np.ones(3) * diameter_mm / spacing
                dx, dy, dz = diameter_pixel
                tol = self.exclude_tol
                mask[z-dz//2*tol:z+dz//2*tol,
                     y-dy//2*tol:y+dy//2*tol,
                     x-dx//2*tol:x+dx*tol] = 0
                if self.debug:
                    exclude_image = image[z-dz//2*tol:z+dz//2*tol,
                                          y-dy//2*tol:y+dy//2*tol,
                                          x-dx//2*tol:x+dx//2*tol]
                    np.save('exclude1_image_%d_%d.npy' % (i, j), exclude_image)
            if self.debug:
                print('sampling from %s' % seriesuid)
            t_z, t_y, t_x = np.where(mask==1)  # tissue mask
            if t_z.size > 0:
                if i == (self.negative_samples_from-1):
                    rvs = np.random.randint(0, t_z.size, last_samples)
                else:
                    rvs = np.random.randint(0, t_z.size, nb_samples_per_scan)
                for rv in rvs:
                    z, y, x = t_z[rv], t_y[rv], t_x[rv]
                    sample = self.get_sample(image, x, y, z)
                    if sample is None:
                        continue
                    samples.append(sample)
                    labels.append(0)
                    # count += 1
                    if self.debug:
                        np.save('tissue_samples_%d_%d.npy' % (i, rv), sample)

        # negative border samples
        nb_samples_per_scan = int(nb_border_samples / self.negative_samples_from)
        last_samples = nb_border_samples - (self.negative_samples_from-1) * nb_samples_per_scan
        for i in range(self.negative_samples_from):
            # randomly pick a CT record
            idx = random.randint(0, len(self.macro_batch)-1)
            seriesuid, data = self.macro_batch[idx]
            spacing = data['spacing']  # in x, y, z order
            origin = data['origin']  # in x, y, z order
            image = data['image']  # in z, y, x order
            mask = data['mask']  # in z, y, x order
            sz, sy, sx = image.shape

            # exclude nodule volume 
            rows = self.rows[self.rows['seriesuid'] == seriesuid]
            for j in range(len(rows)):
                row = rows.iloc[j]
                coord_mm = np.asarray([row['coordX'], row['coordY'], row['coordZ']])
                diameter_mm = row['diameter_mm']
                coord_pixel = (coord_mm - origin) / spacing
                x, y, z = coord_pixel
                diameter_pixel = np.ones(3) * diameter_mm / spacing
                dx, dy, dz = diameter_pixel
                tol = self.exclude_tol
                mask[z-dz//2*tol:z+dz//2*tol,
                     y-dy//2*tol:y+dy//2*tol,
                     x-dx//2*tol:x+dx*tol] = 0
                if self.debug:
                    exclude_image = image[z-dz//2*tol:z+dz//2*tol,
                                          y-dy//2*tol:y+dy//2*tol,
                                          x-dx//2*tol:x+dx//2*tol]
                    np.save('exclude2_image_%d_%d.npy' % (i, j), exclude_image)
            if self.debug:
                print('sampling from %s' % seriesuid)
            b_z, b_y, b_x = np.where(mask==2)  # border mask
            if b_z.size > 0:
                if i == (self.negative_samples_from-1):
                    rvs = np.random.randint(0, b_z.size, last_samples)
                else:
                    rvs = np.random.randint(0, b_z.size, nb_samples_per_scan)
                for rv in rvs:
                    z, y, x = b_z[rv], b_y[rv], b_x[rv]
                    sample = self.get_sample(image, x, y, z)
                    if sample is None:
                        continue
                    samples.append(sample)
                    labels.append(0)    

                    if self.debug:
                        np.save('border_samples_%d_%d.npy' % (i, rv), sample)

        # false sample nodules
        if self.feed_false_samples:
            false_sample_files = glob('%s/*-%s*' % 
                (self.false_sample_dir, self.false_sample_date))
            rv = random.randrange(0, len(false_sample_files))
            false_sample_file = false_sample_files[rv]
            if self.verbose:
                print('extracting false sample nodules from', false_sample_file)
            false_sample_data = np.load(false_sample_file)
            idx = np.where(false_sample_data['false_sample_labels'] == 1)[0]
            if len(idx) > 0:
                rvs = np.random.choice(idx, 
                    replace=False, size=min(len(idx), nb_false_sample_nodules))
                for rv in rvs:
                    samples.append(false_sample_data['false_sample_images'][rv].copy())
                    labels.append(1)    

            # false sample nonnodules
            false_sample_files = glob('%s/*-%s*' % 
                (self.false_sample_dir, self.false_sample_date))
            rv = random.randrange(0, len(false_sample_files))
            false_sample_file = false_sample_files[rv]
            if self.verbose:
                print('extracting false sample nonnodules from', false_sample_file)
            false_sample_data = np.load(false_sample_file)
            idx = np.where(false_sample_data['false_sample_labels'] == 0)[0]
            if len(idx) > 0:
                rvs = np.random.choice(idx, 
                    replace=False, size=min(len(idx), nb_false_sample_nonnodules))
                for rv in rvs:
                    samples.append(false_sample_data['false_sample_images'][rv].copy())
                    labels.append(0)

        # add some samples if not enough
        nb_dummy_samples = self.micro_batch_size - len(samples)
        if self.debug or self.verbose:
            print('generating %d dummy samples' % nb_dummy_samples)
        for i in range(nb_dummy_samples):
            sample = samples[0]
            dummy_sample = np.ones_like(sample) * -1000
            samples.append(dummy_sample)  # fake tissue
            labels.append(0)

        self.macro_batch_count += 1
        if self.debug:
            end = time.time()
            time_cost = end - start
            print('time elapsed loading micro batch: %.2f sec' % time_cost)
        samples = np.asarray(samples, dtype=np.float32)
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


class LUNAEvalInput(object):
    """docstring for LUNAEvalInput"""
    def __init__(self, 
                 data_dir,
                 csv_file=None,
                 batch_size=256,
                 min_nodule=10, 
                 max_nodule=100,
                 grid_size=1,  # sampling grid size, 1 is the most conserved
                 sample_size_xy=48,  # image sample size in x,y dimention
                 sample_size_hz=1,  # half image sample size in z, actual size = 2 * hz + 1
                 verbose=False,
                 debug=False,
                 ):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.min_nodule = min_nodule  # in mm
        self.max_nodule = max_nodule  # in mm
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.sample_size_xy = sample_size_xy
        self.sample_size_hz = sample_size_hz
        self.verbose = verbose
        self.debug = debug
        
        # load and parse annotation csv
        if csv_file is not None:
            df = pd.read_csv(self.csv_file)
            diameters = df['diameter_mm'].values
            valid_idx = np.where((diameters > self.min_nodule) * \
                (diameters < self.max_nodule))[0]
            valid_rows = df.iloc[valid_idx]
            valid_seriesuid = valid_rows['seriesuid'].values
            # remove duplicates
            self.seriesuids = list(set(valid_seriesuid.tolist()))
            self.rows = valid_rows
            if self.verbose:
                print('find %d CT scan to process: \n%s' % 
                    (len(self.seriesuids), str(self.rows)))
        else:
            files = glob('%s/*.npz' % self.data_dir)
            self.seriesuids = []
            for f in files:
                basename = os.path.basename(f)
                self.seriesuids.append(basename.split('.')[0])

        self.scan_count = 0
        self.scan = None
        self.candidate_count = 0
        self.finished = False
        self.batch = []

    def next_scan(self):
        if self.scan_count == len(self.seriesuids):
            return 
        else:
            seriesuid = self.seriesuids[self.scan_count]
            self.seriesuid = seriesuid
            fname = '%s/%s.npz' % (self.data_dir, seriesuid)
            if self.debug or self.verbose:
                print 'loading %s' % fname
            data = np.load(fname)
            spacing = data['spacing']
            origin = data['origin']
            image = data['data']
            mask = data['mask']
            # resampling mask
            sz, sy, sx = mask.shape
            grid_z, grid_y, grid_x = np.meshgrid(
                np.arange(sz, step=self.grid_size),
                np.arange(sy, step=self.grid_size),
                np.arange(sx, step=self.grid_size))
            grid_mask = np.zeros_like(mask, dtype=np.int8)
            grid_mask[grid_z, grid_y, grid_x] = 1
            grid_mask *= (mask>0)
            zs, ys, xs = np.where(grid_mask>0)
            self.nb_candidates = zs.size
            if self.verbose:
                print('grid search candidates: %d' % grid_mask.sum())

            data = {'spacing': spacing,
                    'origin': origin,
                    'image': image,
                    'mask': mask,
                    'grid_mask': grid_mask,
                    'zs': zs,  # candidates coordinates
                    'ys': ys,
                    'xs': xs}
            self.scan = data
            self.scan_count += 1
            self.finished = False
            self.candidate_count = 0

    def next_batch(self):
        size_xy = self.sample_size_xy
        size_hz = self.sample_size_hz
        samples = []
        coords = []

        sz, sy, sx = self.scan['image'].shape
        for i in range(self.batch_size):
            if self.candidate_count >= self.nb_candidates:
                if self.verbose:
                    print('making dummy sample')
                sample = np.ones((size_hz*2+1, size_xy, size_xy)) * -1000  # dummy sample
                samples.append(sample)
                coords.append([-1, -1, -1])
                self.finished = True
            else:
                z = self.scan['zs'][self.candidate_count]
                y = self.scan['ys'][self.candidate_count]
                x = self.scan['xs'][self.candidate_count]
                if (z-size_hz) < 0 or (z+size_hz) > (sz-1) \
                    or (y-size_xy/2) < 0 or (y+size_xy/2) > (sy-1) \
                    or (x-size_xy/2) < 0 or (x+size_xy/2) > (sx-1):  # volume borders
                    sample = np.ones((size_hz*2+1, size_xy, size_xy)) * -1000  # dummy sample
                    if self.verbose:
                        print('making dummy sample')
                else:
                    sample = self.scan['image'][int(z-size_hz):int(z+size_hz+1),
                        int(y-size_xy/2):int(y-size_xy/2+size_xy),
                        int(x-size_xy/2):int(x-size_xy/2+size_xy)]
                samples.append(sample)
                coords.append([z, y, x])
            self.candidate_count += 1
        samples = np.asarray(samples, dtype=np.float32)
        coords = np.asarray(coords, dtype=np.int64)
        self.batch = [coords, samples]
        return self.batch


if __name__ == '__main__':
    train_dir = '/Volumes/SPIDATA/TIANCHI/train_processed'
    csv_file = '/Volumes/SPIDATA/TIANCHI/csv/train/annotations.csv'
    show = True
    train_input = LUNATrainInput(train_dir, csv_file, 
                                 min_nodule=10, 
                                 max_nodule=100, 
                                 debug=False)
    data, labels = train_input.next_micro_batch()
    if show:
        import matplotlib.pyplot as plt

        rv = random.randint(0, len(data))
        plt.imshow(data[rv][0,:,:], cmap='gray')
        plt.clim((-600, 200))
        plt.show()
