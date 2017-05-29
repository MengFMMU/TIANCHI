#!/usr/bin/env python

"""
Usage:
    preprocessing.py <mhd_file_or_dir>  [options]

Options:
    -h --help                                           Show this screen.
    -o --output=output_directory                        Output directory [default: output].
    --threshold=threshold                               Threshold for segmentation [default: -600].
    --new-spacing=new_spacing                           New spacing [default: 1,1,1].
    --tissue-dilation-radius=tissue_dilation_radius     Lung tissue dilation radius [default: 5].
    --tissue-erosion-radius=tissue_erosion_radius       Lung tissue erosion radius [default: 5].
    --border-dilation-radius=border_dilation_radius     Lung border dilation radius [default: 20].
    --border-erosion-radius=border_erosion_radius       Lung border erosion radius [default: 15].
    --border-area-thres=border_area_thres               Border area threhold [default: 400].
    --min-nodule-area=min_nodule_area                   Min nodule area [default: 10].
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology
from skimage.morphology import dilation, erosion, disk
from tqdm import tqdm

from mpi4py import MPI

import os
from glob import glob
from docopt import docopt


def resample(image, old_spacing=None, new_spacing=None):
    old_spacing = np.asarray(old_spacing, dtype=np.float32)
    new_spacing = np.asarray(new_spacing, dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = img_array.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / img_array.shape
    new_spacing = spacing / real_resize_factor
    new_image = ndimage.interpolation.zoom(img_array, real_resize_factor, mode='nearest')
    return new_image, new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > thres, dtype=np.int8)+1
    labels = measure.label(binary_image)
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_slice1 = labels[:,0,:]
    background_slice2 = labels[:,-1,:]
    background_labels = np.unique(background_slice1).tolist() + \
        np.unique(background_slice2).tolist() 
    #Fill the air around the person
    for background_label in background_labels:
        binary_image[labels == background_label] = 2
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    return binary_image


if __name__ == '__main__':
    # parse command options
    argv = docopt(__doc__)
    mhd_file_or_dir = argv['<mhd_file_or_dir>']
    if os.path.isfile(mhd_file_or_dir):
        mhd_files = [mhd_file_or_dir, ]
    elif os.path.isdir(mhd_file_or_dir):
        mhd_files = glob('%s/train*/*.mhd' % mhd_file_or_dir)
    else:
        print('Error! Wrong npz file or directory: %s.' % 
            npz_file_or_dir)
        sys.exit()

    output_dir = argv['--output']
    new_spacing = map(int, argv['--new-spacing'].split(','))
    border_dilation_radius = int(argv['--border-dilation-radius'])
    border_erosion_radius = int(argv['--border-erosion-radius'])
    tissue_dilation_radius = int(argv['--tissue-dilation-radius'])
    tissue_erosion_radius = int(argv['--tissue-erosion-radius'])
    thres = float(argv['--threshold'])  # ROI
    border_area_thres = float(argv['--border-area-thres'])   # borders has 100 pixels at least
    min_nodule_area = float(argv['--min-nodule-area'])

    # MPI setup
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # master sends jobs
    if rank == 0:
        if os.path.isdir(output_dir):
            pass
        else:
            try:
                os.makedirs('%s' %output_dir)
            except Exception as e:
                raise e

        # master send jobs to slaves
        nb_mhd_files = len(mhd_files)
        job_size = nb_mhd_files // size 
        jobs = []
        for i in range(size):
            if i == (size - 1):
                job = np.arange(i*job_size, nb_mhd_files)
            else:
                job = np.arange(i*job_size, (i+1)*job_size)
            jobs.append(job)
            if i == 0:
                continue
            else:
                comm.send(job, dest=i)
        job = jobs[0] 
        # slaves receive jobs
    else:
        job = comm.recv(source=0)

    for i in tqdm(range(len(job))):
        mhd_file = mhd_files[job[i]]
        itk_img = sitk.ReadImage(mhd_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # in z, y, x order
        spacing = np.asarray(itk_img.GetSpacing())[::-1]  # in z, y, x order
        origin = np.array(itk_img.GetOrigin())    

        new_image, new_spacing = resample(img_array, 
            old_spacing=spacing, new_spacing=new_spacing)
        
        segmented_lungs = segment_lung_mask(new_image, False)
        segmented_lungs_filled = segment_lung_mask(new_image, True)
        nodule_mask1 = segmented_lungs_filled - segmented_lungs  # inside lung tissue
        for j in range(len(nodule_mask1)):
            mask_slice = nodule_mask1[j].copy()
            mask_dilation = dilation(mask_slice, disk(tissue_dilation_radius))
            mask_erosion = erosion(mask_dilation, disk(tissue_erosion_radius))
            label = measure.label(mask_erosion, background=0)
            vals, counts = np.unique(label, return_counts=True)
            counts = counts[vals != 0]
            vals = vals[vals != 0]
            for k in range(len(counts)):
                count = counts[k]
                val = vals[k]
                if count < min_nodule_area:
                    mask_erosion[label == val] = 0
            nodule_mask1[j] = mask_erosion

        nodule_mask2 = np.zeros_like(segmented_lungs)  # lung borders
        for j in range(len(segmented_lungs)):
            lung_slice = segmented_lungs[j]
            lung_dilation = dilation(lung_slice, disk(border_dilation_radius))
            lung_erosion = erosion(lung_dilation, disk(border_erosion_radius))
            mask_slice = lung_erosion - lung_slice    

            label = measure.label(mask_slice, background=0)
            vals, counts = np.unique(label, return_counts=True)
            counts = counts[vals != 0]
            vals = vals[vals != 0]
            border_labels = vals[counts > border_area_thres]
            tissue_labels = vals[counts <= border_area_thres]
            for border_label in border_labels:
                mask_slice[label == border_label] = 2  # mask lung border as 2
            for tissue_label in tissue_labels:
                mask_slice[label == tissue_label] = 0  # remove tissue mask
            nodule_mask2[j] = mask_slice
        nodule_mask = nodule_mask1 + nodule_mask2
        # write to file
        np.savez('%s/%s.npz' % (output_dir, os.path.basename(mhd_file)[:-4]),
                data=new_image,  # in z, y, x order
                mask=nodule_mask,  # in z, y, x order
                origin=origin,  # in x, y, z order
                spacing=new_spacing[::-1],  # in x, y, z order
            )
