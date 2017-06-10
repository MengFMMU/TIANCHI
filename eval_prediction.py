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
    pred_df = pd.read_csv(pred_csv_file)
    nb_nodules = len(ref_df)

    ref_probability = np.zeros(len(ref_df), dtype=np.float)
    pred_flag = np.zeros(len(pred_df), dtype=np.int)
    pred_diameter = np.zeros(len(pred_df), dtype=np.float)

    for i in tqdm(range(len(pred_df))):
        pred_row = pred_df.iloc[i]
        seriesuid = pred_row['seriesuid']
        for j in range(len(ref_df)):
            ref_row = ref_df.iloc[j]
            if ref_row['seriesuid'] != seriesuid:
                continue
            diameter = ref_row['diameter_mm']
            probability = pred_row['probability']
            pred_coord = np.array([pred_row['coordX'],
                                  pred_row['coordY'],
                                  pred_row['coordZ']])
            ref_coord = np.array([ref_row['coordX'],
                                 ref_row['coordY'],
                                 ref_row['coordZ']])
            dist = sqrt(((pred_coord - ref_coord)**2.).sum())
            if dist <= diameter/2:
                ref_probability[j] = max(ref_probability[j], probability)
                pred_flag[i] = max(pred_flag[i], 1)
                pred_diameter[i] = diameter

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