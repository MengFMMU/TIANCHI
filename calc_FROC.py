#!/usr/bin/env python

"""
Usage:
    calc_FROC.py <score_volume_dir> [options]

Options:
    -h --help                                       Show this screen.
    --csv-file=csv_file                             Annotation file of nodule coordinates.
    --min-nodule=min_nodule                         Minimum nodule diameter in mm [default: 10].
    --max-nodule=max_nodule                         Maximum nodule diameter in mm [default: 100].
    --smooth-sigma=smooth_sigma                     Gaussian sigma used in smoothing [default: 3].
    --peak-dist=peak_dist                           Peaks are separated by at least min_distance [default: 30].
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
    csv_file = argv['--csv-file']
    min_nodule = float(argv['--min-nodule'])
    max_nodule = float(argv['--max-nodule'])
    smooth_sigma = float(argv['--smooth-sigma'])
    peak_dist = float(argv['--peak-dist'])
    output = argv['--output']

    score_files = glob('%s/*.npz' % score_volume_dir)
    nb_scans = len(score_files)

    if csv_file is not None:
        df = pd.read_csv(csv_file)  

        seriesuids = []
        coords = []
        diameters = []
        nb_nodules = 0
        for f in score_files:
            basename = os.path.basename(f)
            seriesuid = basename.split('.')[0]
            rows = df[df['seriesuid']==seriesuid]
            for i in range(len(rows)):
                row = rows.iloc[i]
                if row['diameter_mm'] > min_nodule \
                    and row['diameter_mm'] < max_nodule:
                    coords.append(row.values[1:-1].astype(np.float))
                    diameters.append(row.values[-1].astype(np.float))
                    seriesuids.append(seriesuid)
                    nb_nodules += 1

        print('found %d nodules in %d scans' % 
            (nb_nodules , nb_scans))

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
            threshold_abs=0.1)

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

    if csv_file is not None:
        # add FLAG whether the nodule is a real nodule
        FLAGs = []
        diameters = []
        for i in range(len(df_predicted)):
            row_values = df_predicted.iloc[i].values
            seriesuid = row_values[0]
            coordX = row_values[1]
            coordY = row_values[2]
            coordZ = row_values[3]
            # compare with real nodules
            rows = df[df['seriesuid'] == seriesuid]
            FLAG = 0
            for j in range(len(rows)):
                row = rows.iloc[j]
                x = row['coordX']
                y = row['coordY']
                z = row['coordZ']
                diameter = row['diameter_mm']
                dist = sqrt((x-coordX)**2 + (y-coordY)**2 + (z-coordZ)**2)
                if dist <= (diameter/2):
                    FLAG = 1
            FLAGs.append(FLAG)
            if FLAG == 1:
                diameters.append(diameter)
            else:
                diameters.append(0.)
        df4 = pd.DataFrame(FLAGs, columns=['FLAG',])
        df5 = pd.DataFrame(diameters, columns=['diameter_mm',])
        df_predicted = df_predicted.join(df4)
        df_predicted = df_predicted.join(df5)

    # sort by probability
    df_predicted.sort_values('probability', ascending=False, inplace=True)
    df_predicted.to_csv(output, index=False)

    if csv_file is not None:
        # calculate FROC curve
        FP_rates = [1./8, 1./4, 1./2, 1., 2., 4]
        sensitivities = []
        for FP_rate in FP_rates:
            TP = 0
            FP = 0
            max_FP = int(FP_rate*len(score_files))
            for i in range(len(df_predicted)):
                FLAG = df_predicted.iloc[i]['FLAG']
                if FLAG == 1:  # hit real nodule
                    TP += 1
                else:
                    FP += 1
                if FP > max_FP:
                    sensitivitiy = float(TP)/float(nb_nodules)
                    sensitivities.append(sensitivitiy)
                    break
        mean_sensitivity = np.mean(sensitivities)
        print('FROC: ', sensitivities)
        print('mean FROC: %.3f' % mean_sensitivity)
        