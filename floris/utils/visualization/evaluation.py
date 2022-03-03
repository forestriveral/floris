import os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

from floris.utils.visualization import property as ppt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     EVALUATION                               #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def array_power_single_direction(data, labels=None, ref=None, ax=None,
                                 ls=None, cl=None):
    if ax is None:
        fig = plt.figure(figsize=(9, 6), dpi=120)
        ax = fig.add_subplot(111)
    # colors = list(mcolors.TABLEAU_COLORS.keys())
    x = np.arange(1, data.shape[1] + 1)
    if labels is None:
        labels = [ i + 1 for i in range(data.shape[0])]
    normal_array_power = data / np.max(data, 1)
    for i in range(data.shape[0]):
        ax.plot(x,
                normal_array_power[i],
                label=labels[i],
                c=ppt.colors[i],
                lw=2,
                linestyle=ppt.lines[i],
                markersize=0,
                marker=ppt.markers[-(i + 1)])
    if ref is not None:
        ppt.colors.insert(0, "w")
        for i, tag in enumerate(ref.columns):
            ax.plot(x,
                    ref[tag].values,
                    label=tag,
                    c='w',
                    lw=0.,
                    linestyle='-',
                    markersize=8,
                    marker=ppt.markers[i],
                    markeredgecolor='k',
                    markeredgewidth=1)
    
    ax.set_xlabel('Turbine Row', ppt.font15)
    ax.set_ylabel('Normalized Power', ppt.font15)
    ax.set_xlim((0., data.shape[1] + 1.))
    ax.set_ylim((np.round(np.min(normal_array_power) * 0.8, 1), 1.1))
    ax.tick_params(labelsize=15, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc="best", prop=ppt.font15, edgecolor='None', frameon=False,
              labelspacing=0.4, bbox_transform=ax.transAxes)
    
    # plt.savefig(f"{}/output/{}.png", format='png', dpi=300, bbox_inches='tight')
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
