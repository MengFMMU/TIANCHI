#!/usr/bin/env python

"""
Usage:
    calc_FROC.py <score_volume_dir> [options]

Options:
    -h --help                                       Show this screen.
    --smooth-sigma=smooth_sigma                     Gaussian sigma used in smoothing [default: 3].
    --peak-dist=peak_dist                           Peaks are separated by at least min_distance [default: 30].
    --peak-thres=peak_thres                         Peak threshold [default: 0.1].
    --output=prediction.csv                         Nodule predicted csv file [default: prediction.csv].
"""

import numpy as np 
import pandas as pd
from math import sqrt

from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from docopt import docopt 
from glob import glob
import os
from tqdm import tqdm

if __name__ == '__main__':
    # parse command options
    argv = docopt(__doc__)
    score_volume_dir = argv['<score_volume_dir>']
    smooth_sigma = float(argv['--smooth-sigma'])
    peak_dist = float(argv['--peak-dist'])
    peak_thres = float(argv['--peak-thres'])
    output = argv['--output']

    score_files = glob('%s/*.npz' % score_volume_dir)
    nb_scans = len(score_files)

    # find possible nodules for each scan
    for i in tqdm(range(len(score_files))):
        f = score_files[i]
        seriesuid = os.path.basename(f).split('.')[0]
        # print('processing %s' % seriesuid)
        data = np.load(f)
        score = data['score_volume']
        origin = data['origin']  # in x, y, z order
        spacing = data['spacing']  # in x, y, z order
        score_smooth = gaussian_filter(score, sigma=smooth_sigma)
        score_smooth -= score_smooth.min()  # make data_smooth 0-based
        _r1 = score_smooth.max() - score_smooth.min()
        _r2 = score.max() - score.min()
        scale_ratio = _r2 / _r1
        score_smooth *= scale_ratio

        peaks = peak_local_max(score_smooth, min_distance=peak_dist, 
            threshold_abs=peak_thres)

        coord = np.zeros_like(peaks, dtype=np.float)
        coord[:,0] = peaks[:,2] * spacing[0] + origin[0]  # coordX
        coord[:,1] = peaks[:,1] * spacing[1] + origin[1]  # coordY
        coord[:,2] = peaks[:,0] * spacing[2] + origin[2]  # coordZ
        possibility = score_smooth[peaks[:,0],
                                   peaks[:,1],
                                   peaks[:,2]]

        df1 = pd.DataFrame([seriesuid] * len(peaks))
        df2 = pd.DataFrame(coord)
        df3 = pd.DataFrame(possibility)
        _df = pd.concat([df1, df2, df3], axis=1)
        if i == 0:
            df_predicted = _df
        else:
            if len(_df) > 0:
                df_predicted = pd.concat([df_predicted, _df], axis=0, ignore_index=True)
            else:
                pass
    df_predicted.columns = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']

    # sort by probability
    df_predicted.sort_values('probability', ascending=False, inplace=True)
    df_predicted.to_csv(output, index=False)