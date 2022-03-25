from __future__ import annotations

# import os
import re
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import OrderedDict
from matplotlib.ticker import MultipleLocator

from floris.utils.visualization import property as ppt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     EVALUATION                               #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class AV_2018_visual(object):

    def __init__(self,):
        pass

    @classmethod
    def show(cls, **kwargs):
        default_figures = {'Fig_4_6': cls.Fig_4_6_plot,
                            'Fig_10_14': cls.Fig_10_14_plot,}
        fig_id = kwargs['eval_data']['config']['fig_id']
        if fig_id in default_figures.keys():
            return default_figures[fig_id](**kwargs)
        else:
            raise ValueError(f"{fig_id} plotting is not supported")

    @classmethod
    def Fig_4_6_plot(cls, ref=True, **kwargs):
        config, power = kwargs['eval_data'].pop('config'), kwargs['eval_data']
        turbine, baseline = kwargs['baseline_data'][0], kwargs['baseline_data'][1]
        turbine_ind = list(turbine) if turbine.ndim < 2 else list(turbine[0])

        labels = list(power.keys())
        colors = list(mcolors.TABLEAU_COLORS.keys())
        # linestyles = ['-', '-', '-', '-', '-']
        array_labels = [str(int(t)) for t in turbine_ind]
        x_range = np.arange(1, len(turbine_ind) + 1)
        y_range = np.arange(0, 1.2, 0.2)

        fig = plt.figure(figsize=(9, 6), dpi=120)
        ax = fig.add_subplot(111)
        for i, case in enumerate(labels):
            array_power = turbine_power_extract(turbine, power[case]['power'])[:, 0, :]
            normal_array_power = array_power / np.max(array_power, axis=1)[:, None]
            ax.plot(x_range,
                    normal_array_power.mean(0),
                    label=labels[i],
                    c=ppt.colors[i],
                    lw=2,
                    linestyle=ppt.lines[i],
                    markersize=0,
                    marker=ppt.markers[-(i + 1)])
        if ref:
            ax.plot(x_range,
                    np.zeros(x_range.shape),
                    label='Obs',
                    c='w',
                    lw=0.,
                    linestyle='-',
                    markersize=8,
                    marker='o',
                    markeredgecolor='k',
                    markeredgewidth=1)

        ax.set_xlabel('Turbine array (ID)', ppt.font18)
        ax.set_xlim((0., len(x_range) + 1.))
        ax.set_xticks(x_range); ax.set_xticklabels(array_labels)
        ax.set_ylabel('Normalized Power', ppt.font18)
        ax.set_ylim((np.round(np.min(normal_array_power) * 0.8, 1), 1.1))
        ax.set_yticks(y_range); ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.tick_params(labelsize=18, colors='k', direction='in',
                       top=True, bottom=True, left=True, right=True)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # ax.xaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.legend(loc="best", prop=ppt.font15, edgecolor='None', frameon=False,
                  labelspacing=0.4, bbox_transform=ax.transAxes)

        # plt.savefig(f"../../outputs/{}.png", format='png', dpi=300, bbox_inches='tight')
        # pd.DataFrame(data.T, columns=legend).to_csv("{}/output/{}.csv".format(file_dir, kwargs["dsave"]))
        plt.show()

    @classmethod
    def Fig_10_14_plot(cls, ref=True, **kwargs):
        config, power = kwargs['eval_data'].pop('config'), kwargs['eval_data']
        turbine_ind = np.array([config['turbine_id']])[None, :]
        direction_range, baseline = list(kwargs['baseline_data'][0][0]), kwargs['baseline_data'][1]

        labels = list(power.keys())
        colors = list(mcolors.TABLEAU_COLORS.keys())
        # linestyles = ['-', '-', '-', '-', '-']
        # array_labels = [str(int(t)) for t in turbine_ind]
        # x_range = np.arange(1, len(turbine_ind) + 1)
        y_range = np.arange(0, 1.2, 0.2)

        fig = plt.figure(figsize=(9, 6), dpi=120)
        ax = fig.add_subplot(111)
        for i, case in enumerate(labels):
            array_power = turbine_power_extract(
                turbine_ind, power[case]['power'], mean=False)[0, :, 0, 0]
            normal_array_power = array_power / np.max(array_power)
            x_range = power[case]['wind_directions']
            ax.plot(x_range,
                    normal_array_power,
                    label=labels[i],
                    c=ppt.colors[i],
                    lw=2,
                    linestyle=ppt.lines[i],
                    markersize=0,
                    marker=ppt.markers[-(i + 1)])
        if ref:
            ax.plot(x_range,
                    np.zeros(x_range.shape),
                    label='Obs',
                    c='w',
                    lw=0.,
                    linestyle='-',
                    markersize=8,
                    marker='o',
                    markeredgecolor='k',
                    markeredgewidth=1)

        ax.set_xlabel('Turbine array (ID)', ppt.font18)
        ax.set_xlim([direction_range[0], direction_range[1] + 1])
        # ax.set_xticks(x_labels); ax.set_xticklabels(array_labels)
        ax.set_ylabel('Normalized Power', ppt.font18)
        ax.set_ylim((np.round(np.min(normal_array_power) * 0.8, 1), 1.1))
        ax.set_yticks(y_range); ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.tick_params(labelsize=18, colors='k', direction='in',
                       top=True, bottom=True, left=True, right=True)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        ax.xaxis.set_major_locator(MultipleLocator(5))
        # ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.legend(loc="best", prop=ppt.font15, edgecolor='None', frameon=False,
                  labelspacing=0.4, bbox_transform=ax.transAxes)

        # plt.savefig(f"../../outputs/{}.png", format='png', dpi=300, bbox_inches='tight')
        # pd.DataFrame(data.T, columns=legend).to_csv("{}/output/{}.csv".format(file_dir, kwargs["dsave"]))
        plt.show()


def farm_power_all_direction(data, labels=None, ref=None, ax=None,
                             ls=None, cl=None):
    if ax is None:
        fig = plt.figure(figsize=(9, 6), dpi=120)
        ax = fig.add_subplot(111)
    # colors = list(mcolors.TABLEAU_COLORS.keys())
    # for i in range(data.shape[0]):
    #     winds, label = wind_range_from_label(legend[i])
    #     ax.plot(winds, data[i, :], color=ppt.colors[i], linewidth=1.5,
    #             linestyle=ppt.lines[i], label=label)
    # winds = wind_range_from_label(legend[0])[0]
    # title = 'Distribution of the normalized Horns Rev wind farm power ouput'
    # plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def turbine_power_extract(turbine_ind, case_power, mean=True):
    assert isinstance(turbine_ind, np.ndarray) and isinstance(case_power, np.ndarray)
    turbine_power = np.zeros((turbine_ind.shape[0], case_power.shape[0],
                              case_power.shape[1], turbine_ind.shape[1]))
    for i in range(turbine_ind.shape[0]):
        turbine_power[i] = case_power[:, :, turbine_ind[i] - 1]
    if mean:
        turbine_power = np.mean(turbine_power, axis=0)
    return turbine_power


def wind_range_from_label(label):
    s, e = re.search(r"(\([^\)]+\))", label).span()
    w = label[s: e + 1] if s == 0 else label[s - 1: e]
    label = label.replace(w, "")
    wind_range = eval(w.strip("+"))
    return np.arange(wind_range[0], wind_range[1]), label


def turbine_array_power(inds, powers):
    if inds.ndim == 2:
        array_power = np.zeros((powers.shape[0], inds.shape[1]))
        tmp = np.zeros(inds.shape)
        for j in range(powers.shape[0]):
            for i in range(inds.shape[0]):
                tmp[i, :] = powers[j, :][inds[i, :] - 1]
            array_power[j, :] = np.mean(tmp, axis=0)
        return array_power
    elif inds.ndim == 1:
        return powers[:, inds - 1]
    else:
        raise ValueError("Invalid index of turbine array!")



if __name__ == "__main__":
    pass
