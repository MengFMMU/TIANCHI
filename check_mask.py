#!/usr/bin/env python

"""
Usage:
    check_mask.py <npz_file_or_dir> <csv_file> [options]

Options:
    -h --help                                        Show this screen.
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

    if os.path.isfile(npz_file_or_dir):
        npz_files = npz_file_or_dir
    elif os.path.isdir(npz_file_or_dir):
        npz_files = glob('%s/*.npz' % npz_file_or_dir)
    else:
        print('Error! Wrong npz file or directory: %s.' % 
            npz_file_or_dir)
        sys.exit()

    # load csv file
    df = pd.read_csv(csv_file)

    # check mask 
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
        mask = data['mask']  # in z, y, x order
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
