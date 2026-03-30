#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Florent Meyniel, Audrey Mazancieux

This script performs the second-level analysis for the probability learning task of 
the NACONF dataset.
It uses the output of the first-level analysis, compute stats, and plot results.

The code has been adapted to work with the Neuromod project

"""

import os
import pickle
from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from params_and_paths import Paths, Params

paths = Paths()
params = Params()

OUT_DIR = '/home_local/alice_hodapp/NeuroModAssay/domain_general/behavior'

N_BINS = 6
FONTSIZE = 16

with open(os.path.join(OUT_DIR, 'correlations_IO_all_sub.pickle'), 'rb') as f:
    corr_dict = pickle.load(f)

with open(os.path.join(OUT_DIR, 'NAConf_behav_summary.txt'), "w") as file:

    # group-level test
    for name in ['r_prob', 'r_conf', 'res_r_conf']:
        mean = np.mean(np.fromiter(corr_dict[name].values(), dtype=float))
        std = np.std(np.fromiter(corr_dict[name].values(), dtype=float))
        ttest = ttest_1samp(np.fromiter(corr_dict[name].values(), dtype=float), 0)
        file.write(f"{name}: mean={mean}, SD={std}, t={ttest.statistic}, df={ttest.df}, p={ttest.pvalue}")
        file.write('\n\n')
        plt.figure()
        plt.plot(np.zeros(len(np.fromiter(corr_dict[name].values(), dtype=float))),
                np.fromiter(corr_dict[name].values(), dtype=float), '.')
        plt.ylabel(name)
        fname = f'NAConf_behav_{name}.png' 
        plt.savefig(os.path.join(OUT_DIR, fname))



