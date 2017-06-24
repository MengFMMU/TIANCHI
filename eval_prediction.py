#!/usr/bin/env python

"""
Usage:
    eval_prediction.py <ref_csv> <pred_csv> <nb_scan> [options]

Options:
    -h --help                                       Show this screen.
    -o --output=evaluation.xlsx                     Evaluation output [default: evaluation.xlsx].
"""

import numpy as np 
import pandas as pd
from math import sqrt

from docopt import docopt 
from tqdm import tqdm


if __name__ == '__main__':
    # parse command options
    argv = docopt(__doc__)
    ref_csv_file = argv['<ref_csv>']
    pred_csv_file = argv['<pred_csv>']
    nb_scan = int(argv['<nb_scan>'])
    output = argv['--output']

    ref_df = pd.read_csv(ref_csv_file)
    ref_coords = np.zeros((len(ref_df), 3))
    ref_coords[:,0] = ref_df['coordX'].values  
    ref_coords[:,1] = ref_df['coordY'].values  
    ref_coords[:,2] = ref_df['coordZ'].values 
    ref_diameters = ref_df['diameter_mm'].values
    ref_seriesuids = ref_df['seriesuid'].values

    pred_df = pd.read_csv(pred_csv_file)
    nb_nodules = len(ref_df)

    ref_probability = np.zeros(len(ref_df), dtype=np.float)
    pred_flag = np.zeros(len(pred_df), dtype=np.int)
    pred_diameter = np.zeros(len(pred_df), dtype=np.float)

    found_nodule_ids = []
    for i in tqdm(range(len(pred_df))):
        pred_row = pred_df.iloc[i]
        seriesuid = pred_row['seriesuid']
        idx = np.where(ref_seriesuids == seriesuid)[0]

        pred_coord = np.array([pred_row['coordX'],
                                  pred_row['coordY'],
                                  pred_row['coordZ']])
        ref_coords_valid = ref_coords[idx]
        ref_diameters_valid = ref_diameters[idx]
        dists = np.sqrt(((pred_coord - ref_coords_valid)**2.).sum(axis=1))
        nodule_idx = idx[np.where(dists <= ref_diameters_valid/2)[0]]
        if len(nodule_idx) > 0:
            nodule_id = nodule_idx[0]
            ref_probability[nodule_id] = max(pred_row['probability'], 
                ref_probability[nodule_id])
            pred_diameter[i] = ref_diameters[nodule_id]
            if nodule_id in found_nodule_ids:
                pred_flag[i] = -1
                print('found duplicate nodule at %ith row' % i)
            else:
                pred_flag[i] = 1
                found_nodule_ids.append(nodule_id)

    ref_df = ref_df.join(pd.DataFrame(ref_probability, columns=['probability']))
    pred_df = pred_df.join(pd.DataFrame(pred_flag, columns=['FLAG']))   
    pred_df = pred_df.join(pd.DataFrame(pred_diameter, columns=['diameter']))   

    # calculate FROC curve
    FP_rates = [1./8, 1./4, 1./2, 1., 2., 4., 8., 16., 32., 64., 128., 256]
    sensitivities = []
    for FP_rate in FP_rates:
        TP = 0
        FP = 0
        max_FP = int(FP_rate*nb_scan)
        for i in range(len(pred_df)):
            FLAG = pred_df.iloc[i]['FLAG']
            if FLAG == 1:  # hit real nodule
                TP += 1
            elif FLAG == -1:  # duplicate nodule
                continue
            else:
                FP += 1
            if FP > max_FP or (i == (len(pred_df)-1)):
                sensitivitiy = float(TP)/float(nb_nodules)
                sensitivities.append(sensitivitiy)
                break
    FROC_df = pd.DataFrame(FP_rates, columns=['FP ratio'])
    FROC_df = FROC_df.join(pd.DataFrame(sensitivities, columns=['sensitivity']))
    sensitivities = np.array(sensitivities)
    mean_sensitivity = sensitivities[0:7].mean()  # mean value of first 7 sensitivity
    print('mean FROC: %.3f' % mean_sensitivity)

    # write to output
    writer = pd.ExcelWriter(output)
    ref_df.to_excel(writer,'reference')
    pred_df.to_excel(writer,'prediction')
    FROC_df.to_excel(writer, 'FROC')
    writer.save()