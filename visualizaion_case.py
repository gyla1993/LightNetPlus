# Case visualization for every test period.

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from global_var import LIG_file_dir, result_file_dir, visualization_file_dir

def plot_one(y_true, y_pred, plt_name, save_path):
    truth_indices = np.argwhere(y_true >= 0.5)
    pred_indices = np.argwhere(y_pred >= 0.5)
    pred_hit_indices = np.argwhere(np.logical_and(y_true >= 0.5, y_pred >= 0.5))

    fig = plt.figure(figsize=(7.0, 7.0))
    ax = fig.add_subplot(111)
    sz = 16
    ax.scatter(truth_indices[:, 0], truth_indices[:, 1], c='b', s=sz, linewidths=0)
    ax.scatter(pred_indices[:, 0], pred_indices[:, 1], c='g', s=sz, linewidths=0)
    ax.scatter(pred_hit_indices[:, 0], pred_hit_indices[:, 1], c='r', s=sz, linewidths=0)
    # ax.set_title(plt_name)
    ax.set_xlim([-1, 160])
    ax.set_ylim([-1, 160])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(save_path+plt_name+'.png', bbox_inches='tight', pad_inches=0.01)
    plt.close()

def plot_one_heat(y_true, y_pred, plt_name, save_path):
    y_pred = np.transpose(y_pred)
    norm = Normalize(vmin=0.0, vmax=1.0)
    fig = plt.figure(figsize=(7.0, 7.0))
    ax = fig.add_subplot(111)
    ax.imshow(y_pred, norm=norm, cmap=plt.cm.get_cmap('viridis'))
    # print(y_pred)
    # ax.imshow(y_pred)
    # ax.set_title(plt_name)
    ax.set_xlim([-1, 160])
    ax.set_ylim([-1, 160])
    plt.savefig(save_path+'heat_'+plt_name+'.png', bbox_inches='tight', pad_inches=0.01)
    plt.close()

def plot_case(model_list, start_time, truth_path, pred_path_, save_path):
    n_model = len(model_list)

    scale = 4
    time_step = 6
    n_obs = 1
    fig = plt.figure(figsize=(time_step*scale, (n_model+n_obs)*0.775*scale))
    gs = gridspec.GridSpec(n_model+n_obs, time_step)
    gs.update(wspace=0.01, hspace=0.01, top=0.95, bottom=0.05, left=0.17, right=0.845)  # set the spacing between axes.
    # fig, axs = plt.subplots(gs)
    delta_list = [0, 1, 2, 3, 4, 5]
    now = datetime.datetime.strptime(start_time, '%Y%m%d%H%M')

    sz = 12
    for ind in range(6):
        delta = delta_list[ind]
        dt = now + datetime.timedelta(hours=delta)
        dt_str = dt.strftime('%Y%m%d%H%M')
        truth_tar = truth_path + dt_str + '_truth'

        if not os.path.exists(truth_tar):
            plt.close()
            return

        with open(truth_tar) as f:
            y_true = np.array(f.readlines(), dtype=float)
        y_true = y_true.reshape((159, 159))
        y_true = np.transpose(y_true)
        truth_indices = np.argwhere(y_true >= 0.5)
        ax = plt.subplot(gs[n_obs-1, ind])
        ax.scatter(truth_indices[:, 0], truth_indices[:, 1], c='r', s=sz, linewidths=0)
        # ax.set(xlim=[-1, 160], ylim=[-1, 160])
        ax.set(xlim=[-1, 160], ylim=[-1, 160], aspect=1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.margins(x=0, y=0)
        # axs[0, ind].set_xticks([])
        # axs[0, ind].set_yticks([])

    norm = Normalize(vmin=0.0, vmax=1.0)
    for model_ind in range(len(model_list)):
        model = model_list[model_ind]
        pred_path = pred_path_+model+'/'
        for ind in range(6):
            delta = delta_list[ind]
            dt = now + datetime.timedelta(hours=delta)
            dt_str = dt.strftime('%Y%m%d%H%M')
            pred_tar = pred_path + dt_str + ('_h%d' % delta)

            if not os.path.exists(pred_tar):
                plt.close()
                return

            with open(pred_tar) as f:
                y_pred = np.array(f.readlines(), dtype=float)
            y_pred = y_pred.reshape((159, 159))
            # y_pred = np.transpose(y_pred)
            ax = plt.subplot(gs[model_ind + n_obs, ind])
            im = ax.imshow(y_pred, alpha=0.9, norm=norm, cmap=plt.cm.get_cmap('jet'))
            # ax.imshow(y_pred, norm=norm, cmap=plt.cm.get_cmap('Wistia'))
            cs = ax.contour(y_pred,  norm=norm, levels=[0.5], linewidths=1, colors='k', linestyles='dashed')
            ax.clabel(cs, inline=1, fmt='%.1f', fontsize=10)
            # ax.imshow(y_pred, cmap=plt.cm.get_cmap('hsv'))
            ax.set_xlim([-1, 160])
            ax.set_ylim([-1, 160])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.margins(x=0, y=0)
            # axs[model_ind + 1, ind].set_xticks([])
            # axs[model_ind + 1, ind].set_yticks([])
    # plt.subplots_adjust(hspace=.001)
    # plt.savefig(save_path+start_time+'.png')
    # cbar_ax = fig.add_axes([0.01, 0.98, 0.5, 0.025])
    # cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    # cbar.ax.tick_params(labelsize=25)
    plt.savefig(save_path+start_time+'.png', bbox_inches='tight', pad_inches=0.02)
    # plt.savefig(save_path + start_time + '.png', bbox_inches='tight', pad_inches=0.01)
    plt.close()

if __name__ == "__main__":

    model_list = [
        'LightNetPlus_WRF-21',
        'LightNetPlus_LIG-18',
        'StepDeep-19',
        'LightNetPlus-09'
    ]

    start_time_list = []
    st = datetime.datetime(2017, 8, 1, 2)
    et = datetime.datetime(2017, 9, 30, 20)
    tt = st
    while tt <= et:
        start_time_list.append(tt.strftime('%Y%m%d%H')+'00')
        tt += datetime.timedelta(hours=6)

    # start_time_list = [
    #     '201708052000',
    #     '201708080800',
    #     '201708081400',
    #     '201708082000',
    #     '201708111400',
    #     '201708120800',
    #     '201708160800',
    #     '201709031400',
    #     '201709211400',
    # ]

    for start_time in start_time_list:
        plot_case(model_list, start_time, LIG_file_dir, result_file_dir, visualization_file_dir)
