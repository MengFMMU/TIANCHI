import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology
from skimage.morphology import dilation, erosion, disk
from tqdm import tqdm

import os
from glob import glob


mhd_files = glob('/Volumes/SPIDATA/TIANCHI/train*/*.mhd')
output_dir = '/Volumes/SPIDATA/TIANCHI/train_processed'
new_spacing = [1, 1, 1]
dilation_radius = 20
erosion_radius = 10
thres = -600  # ROI
border_area_thres = 400   # borders has 100 pixels at least


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
    background_label = labels[0,0,0]
    #Fill the air around the person
    binary_image[background_label == labels] = 2
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


for i in tqdm(range(len(mhd_files))):
    mhd_file = mhd_files[i]
    itk_img = sitk.ReadImage(mhd_file)
    img_array = sitk.GetArrayFromImage(itk_img)  # in z, y, x order
    spacing = np.asarray(itk_img.GetSpacing())[::-1]  # in z, y, x order
    origin = np.array(itk_img.GetOrigin())

    new_image, new_spacing = resample(img_array, 
        old_spacing=spacing, new_spacing=new_spacing)
    lung_mask = segment_lung_mask(new_image, False)

    # split mask into 1(small blobs) and 2(borders)
    nodule_mask = np.zeros_like(lung_mask)
    for j in range(len(lung_mask)):
        lung_slice = lung_mask[j]
        lung_dilation = dilation(lung_slice, disk(dilation_radius))
        lung_erosion = erosion(lung_dilation, disk(erosion_radius))
        nodule_slice = lung_erosion - lung_slice

        label = measure.label(nodule_slice, background=0)
        vals, counts = np.unique(label, return_counts=True)
        counts = counts[vals != 0]
        vals = vals[vals != 0]
        border_labels = vals[counts > border_area_thres]
        for border_label in border_labels:
            nodule_slice[label == border_label] = 2  # mask lung border as 2
        nodule_mask[j] = nodule_slice
    # write to file
    np.savez('%s/%s.npz' % (output_dir, os.path.basename(mhd_file)[:-4]),
            data=new_image,  # in z, y, x order
            mask=nodule_mask,  # in z, y, x order
            origin=origin,  # in x, y, z order
            spacing=new_spacing[::-1],  # in x, y, z order
        )
