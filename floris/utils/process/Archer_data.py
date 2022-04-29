from __future__ import annotations

# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

from floris.utils.visual import property as ppt


data_path = '../data/others/wake_performance_data'
turbine_id_dict = {'An_179': [32, 33, 34, 35, 36],
                   'An_183': [1, 2, 3, 4, 5],
                   'An_228': [1, 31, 32, 43, 44, 45, 66, 67, 68, 77, 78, 79, 85, 86],
                   'An_341': [111, 110, 109, 108, 107, 106],
                   'Lill_180': [15, 22, 28, 39, 43, 46],
                   'Lill_222': [23, 22, 21, 20, 19, 18, 17, 16],
                   'Lill_255': [36, 28, 20, 11, 3],
                   'Lill_300': [46, 42, 37, 32, 25, 17, 9, 2],
                   'NH_260': [1, 2, 3, 4, 5],
                   'Nr_75': [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                   'Nr_255': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                   'Nys_278': [1, 2, 3, 4, 5, 6, 7, 8],
                   'Rd_304': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                   }
model_label = ['Jensen', 'Frandsen', 'Multizone',
               'Turbopark', 'Bastankhah', 'Ishihara-Qian']



def wake_peformance_plot(fname, **kwargs):
    turbine_ind = turbine_id_dict[fname]
    power_data = pd.read_csv(f'{data_path}/{fname}.csv')
    obs_error_up = power_data.pop('obs_up').values
    obs_error_down = power_data.pop('obs_down').values
    print(power_data.columns[2])

    color = ['w', 'k', 'g', 'y', 'purple', 'b', 'r']
    linestyle = ['-', '-', '-', '--', '--', '-', '-']
    x_label = [str(t) for t in turbine_ind]
    x_range = np.arange(len(turbine_ind)) + 1
    line_label = ['Obs', 'Jensen', 'Frandsen', 'Multizone',
                  'Turbopark', 'Bastankhah', 'Ishihara-Qian']

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111)
    min_val = 10.
    for i, label in enumerate(line_label):
        col_name = power_data.columns[i]
        col_data = power_data[col_name].values
        min_val = np.min(col_data) if np.min(col_data) <= min_val else min_val
        if col_name == 'obs_mean':
            ax.plot(x_range,
                    col_data,
                    label=label,
                    c=color[i],
                    lw=0.,
                    linestyle='-',
                    markersize=10,
                    marker='o',
                    markeredgecolor='k',
                    markeredgewidth=1)
        else:
            ax.plot(x_range,
                    col_data,
                    label=label,
                    c=color[i],
                    lw=2.,
                    linestyle=linestyle[i],
                    markersize=0,
                    marker='')

    ax.set_xlabel('Turbine array (ID)', ppt.font18)
    ax.set_xlim((0., len(x_range) + 1.))
    ax.set_xticks(x_range)
    ax.set_xticklabels(x_label)
    ax.set_ylabel('Normalized Power', ppt.font18)
    ax.set_ylim((min_val * 0.8, 1.1))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    direction_label = fname.split('_')[-1]
    ax.text(0.5, 0.91, f'{direction_label} degree', va='top', ha='center',
            fontdict=ppt.font18t, transform=ax.transAxes, )
    ax.tick_params(labelsize=18, colors='k', direction='in',
                    top=True, bottom=True, left=True, right=True)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc="best", prop=ppt.font13, edgecolor='None', frameon=False,
              labelspacing=0.4, bbox_transform=ax.transAxes)
    plt.savefig(f"{data_path}/{fname}_power.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def wake_error_plot():
    fnames = turbine_id_dict.keys()
    power_errors = np.zeros((6, len(fnames)))
    for i, fname in enumerate(fnames):
        power_data = pd.read_csv(f'{data_path}/{fname}.csv')
        obs_mean = power_data.pop('obs_mean').values
        _ = power_data.pop('obs_up').values
        _ = power_data.pop('obs_down').values
        for j, label in enumerate(model_label):
            col_name = power_data.columns[j]
            col_data = power_data[col_name].values
            power_errors[j, i] = np.mean(np.abs(col_data - obs_mean))
    power_errors = np.mean(power_errors, axis=1) * 2.
    print(power_errors)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111)
    # ax.bar(x + i * bar_width, v1,
    #        bar_width,
    #        color=c1,
    #        align='center',
    #        label=labels[i],
    #        hatch=h1,
    #        linewidth=1.0,
    #        edgecolor='k',
    #        alpha=a1,)
    # ticks = ax.get_xticklabels() + ax.get_yticklabels()
    # [tick.set_fontname('Times New Roman') for tick in ticks]
    # ax.tick_params(labelsize=15, colors='k', direction='in',
    #             top=True, bottom=True, left=True, right=True)
    # ax.set_xlabel(r'Wind scenario', ppt.font15)
    # ax.set_xticks(x + (n - 1) * bar_width / 2)
    # ax.set_xticklabels(tick_labels)
    # ax.set_xticklabels([])
    # ax.set_ylim(ylims[j])
    # ax.set_ylabel(ylabels[j], ppt.font15)
    # ax.text(2.2, text_yaxis[j], f"${numbers[j]}$", fontsize=15, color='k')

    plt.show()


if __name__ == "__main__":
    wake_error_plot()
    # wake_peformance_plot('An_183')
    # for fname in turbine_id_dict.keys():
    #     wake_peformance_plot(fname)
