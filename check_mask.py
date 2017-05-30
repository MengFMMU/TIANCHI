#!/usr/bin/env python

"""
Usage:
    check_mask.py <npz_file_or_dir> <csv_file> [options]

Options:
    -h --help                                       Show this screen.
    -o --output=output_file                         Output filename of checking results [default: mask.txt].
"""

import numpy as np 
import pandas as pd

import sys
import os
from glob import glob
from tqdm import tqdm

from docopt import docopt


if __name__ == '__main__':
    # parse command options
    argv = docopt(__doc__)
    npz_file_or_dir = argv['<npz_file_or_dir>']
    csv_file = argv['<csv_file>']
    output = argv['--output']

    if os.path.isfile(npz_file_or_dir):
        npz_files = [npz_file_or_dir, ]
    elif os.path.isdir(npz_file_or_dir):
        npz_files = glob('%s/*.npz' % npz_file_or_dir)
    else:
        print('Error! Wrong npz file or directory: %s.' % 
            npz_file_or_dir)
        sys.exit()

    # load csv file
    df = pd.read_csv(csv_file)

    # check mask 
    nodule_records = []
    missed_nodules = []
    tissue_nodules = []
    border_nodules = []
    nb_files = len(npz_files)

    for i in tqdm(range(len(npz_files))):
        npz_file = npz_files[i]
        # print('processing %d/%d: %s' % 
        #     (i+1, nb_files, npz_file))
        seriesuid = os.path.basename(npz_file).split('.')[0]
        nodules = df[df['seriesuid'] == seriesuid]

        # load CT data
        data = np.load(npz_file)
        origin = data['origin']  # in x, y, z order
        spacing = data['spacing']  # in x, y, z order
        img_array = data['data']  # CT array
        mask = data['mask']  # in z, y, x order
        sx, sy, sz = mask.shape
        mask_ratio = float((mask > 0).sum()) / float(mask.size)

        # process each nodule
        for j in range(len(nodules)):
            nodule = nodules.iloc[j]
            coordX = nodule['coordX']
            coordY = nodule['coordY']
            coordZ = nodule['coordZ']
            diameter = nodule['diameter_mm']

            coord_mm = np.array([coordX, coordY, coordZ])
            coord_pixel = ((coord_mm - origin) / spacing).astype(np.int16)
            x, y, z = coord_pixel
            if min(x, y, z) < 0 or min(sx-1-x, sy-1-y, sz-1-z) < 0:
                mask_ = -1  # mask type
                ct_ = 10000  # CT value
            else:
                mask_ = mask[x, y, z]
                ct_ = img_array[x, y, z] 
            nodule_records.append([int(seriesuid.split('-')[-1]),  # patient id
                coordX, coordY, coordZ,  # nodule coordinates in physical world
                x, y, z,  # nodule coordinates in array
                mask_,  # mask type
                ct_,  # CT value 
                ])
            if min(x, y, z) < 0:
                print('ERROR! Negative coordinates, maybe wrong label?')
            elif mask[z, y, x] == 0:
                missed_nodules.append({'nodule': nodule,
                    'mask ratio': mask_ratio})
            elif mask[z, y, x] == 1:
                tissue_nodules.append({'nodule': nodule,
                    'mask ratio': mask_ratio})
            elif mask[z, y, x] == 2:
                border_nodules.append({'nodule': nodule,
                    'mask ratio': mask_ratio})
            else:
                print('WARNING!!! Something wrong?!')

        np.savetxt(output, nodule_records, fmt='%05d %7.2f %7.2f %7.2f %5d %5d %5d %3d %5d')
