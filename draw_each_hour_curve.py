# This file draws performance curves in hours.

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from matplotlib import cm
from global_var import score_file_dir, curve_file_dir



if __name__ == "__main__":
    draw_comparision = True
    draw_data_sources = False

    if draw_comparision:
        model_list = [
            'PR92',
            'F1',
            'F2',
            'StepDeep-19',
            'LightNetPlus-09'
        ]
        name_list = [
            'PR92',
            'F1',
            'F2',
            'StepDeep',
            'LightNet+'
        ]
        score_list = ['POD', 'FAR', 'TS', 'ETS']
        marker_list = ['o', 's', '*', '^', 'D']
        myColors = cm.Set1(range(6))
        choice = [1, 2, 4, 3, 0]
        val_arr = np.zeros(shape=(len(model_list), len(score_list), 6), dtype=float)

        for i in range(len(model_list)):
            fn = score_file_dir+model_list[i]+'/score_each_hour.txt'
            val_arr[i, :, :] = np.loadtxt(fn)
        scale = 0.8
        x = np.arange(0.5, 5.6, 1.0)
        for j in range(len(score_list)):
            fig = plt.figure(figsize=(8 * scale, 6 * scale))
            for i in range(len(model_list)):
                plt.plot(x, val_arr[i, j, :], marker=marker_list[i], markersize=10, fillstyle='none', ls='--', lw=2.5,
                         color=myColors[choice[i]])
            plt.title(score_list[j], fontsize=20)
            plt.legend(name_list, loc='best', fontsize=14)
            plt.savefig(curve_file_dir + 'comparision_' + score_list[j] + '.pdf')
            plt.close()

    if draw_data_sources:
        model_list = [
            'LightNetPlus_WRF-21',
            'LightNetPlus_LIG-18',
            'LightNetPlus_AWS-24',
            'LightNetPlus_WRF_LIG-09',
            'LightNetPlus_WRF_AWS-13',
            'LightNetPlus_LIG_AWS-24',
            'LightNetPlus-09'
        ]

        name_list = [
            'WRF',
            'LIG',
            'AWS',
            'WRF+LIG',
            'WRF+AWS',
            'LIG+AWS',
            'ALL',
        ]
        score_list = ['POD', 'FAR', 'TS', 'ETS']
        marker_list = ['o', 's', '*', '^', 'v', 'p', 'D']
        myColors = cm.Set1(range(8))
        choice = [1, 2, 3, 4, 6, 7, 0]
        val_arr = np.zeros(shape=(len(model_list), len(score_list), 6), dtype=float)


        for i in range(len(model_list)):
            fn = score_file_dir+model_list[i]+'/score_each_hour.txt'
            val_arr[i, :, :] = np.loadtxt(fn)
        scale = 0.8
        x = np.arange(0.5, 5.6, 1.0)
        for j in range(len(score_list)):
            fig = plt.figure(figsize=(8*scale, 6*scale))
            for i in range(len(model_list)):
                plt.plot(x, val_arr[i, j, :], marker=marker_list[i], markersize=10, fillstyle='none', ls='--', lw=2, color=myColors[choice[i]])
            plt.title(score_list[j], fontsize=20)
            plt.legend(name_list, loc='best', fontsize=14, ncol=2)
            plt.savefig(curve_file_dir + 'data_source_' + score_list[j] + '.pdf')
            plt.close()

