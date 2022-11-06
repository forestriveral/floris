import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

from floris.utils.tools import power_calc_ops_old as power_ops

root = os.path.dirname(os.path.dirname(__file__))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                              PLOT_CONFIGURATION                              #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 15}

font2 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 10}

font3 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 13}

font4 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 15,
         'color': 'b',}

font5 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 20}

line_styles = ["-", "--", "-.", ":", "-", "--", ]

color_group = ['k', 'b', 'y', 'r', 'g', ]

line_maker_style = dict(linestyle=":",
                        linewidth=2,
                        color="maroon",
                        markersize=15)

marker_styles = ["o", "D", "$\diamondsuit$",
                 "+", "x", "s", "^",  "d", ">", "<", "v"]

marker_sizes = [5, 5, 5, 5]

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     VALIDATION                               #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def layout_plot(coordinates, annotate=False):
    x, y = coordinates[:, 0], coordinates[:, 1]
    plt.figure(dpi=100)
    plt.scatter(x, y)
    if annotate:
        num_labs = [str(i) for i in range(1, 81)]
        for i in range(len(num_labs)):
            plt.annotate(num_labs[i], xy=(x[i], y[i]),
                         xytext=(x[i] + 50, y[i] + 50))
    # plt.xlim((-8 * 80., 80 * 80.));plt.ylim((-4 * 80., 70 * 80.))
    plt.show()


def wt_power_eval(legend, data, **kwargs):
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    x = np.arange(1, data.shape[1] + 1)
    # if kwargs.get("ref", False):
    if kwargs.__contains__("ref"):
        for i, tag in enumerate(kwargs["ref"].columns):
            if tag.split("+")[-1] in ["OBS", ]:
                continue
            tag_i = tag.split('+')[-1]
            ax.plot(x, kwargs["ref"][tag].values,
                    linestyle=line_styles[i],
                    c='w',
                    lw=0.00,
                    label=tag_i,
                    markersize=10,
                    marker=marker_styles[i],
                    markeredgecolor='k',
                    markeredgewidth=1.5)
    for i in range(data.shape[0]):
        legend_ii = legend[i].split('+')[-2:]
        legend_i = ['BP' if i == 'Bastankhah' else i for i in legend_ii]
        # legend_i = [legend_i[0], legend_i[2], legend_i[1]]
        ax.plot(x, power_ops.normalized_wt_power(data[i, :]),
                color=color_group[i], linewidth=2,
                linestyle=line_styles[i],
                markersize=0,
                marker=marker_styles[-(i + 1)],
                label=' + '.join(legend_i))
    title = ""
    ax = general_axes_property(ax, 'Turbine Row', 'Normalized Power',
                               (0.5, data.shape[1] + 0.5), (0.1, 1.05),
                               1, title)
    if kwargs.get("psave", False):
        plt.savefig("{}/output/{}.png".format(root, kwargs["psave"]), format='png',
                    dpi=300, bbox_inches='tight')
        print("** Picture {} Save Done !** \n".format(kwargs["psave"]))
    if kwargs.get("dsave", False):
        pd.DataFrame(data.T, columns=legend).to_csv(
            "{}/output/{}.csv".format(root, kwargs["dsave"]))
        print("** Data {} Save Done !** \n".format(kwargs["dsave"]))
    if kwargs.get("show", True):
        plt.show()


def wf_power_eval(legend, data, **kwargs):
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i in range(data.shape[0]):
        winds, label = wind_range_from_label(legend[i])
        ax.plot(winds, data[i, :], color=colors[i], linewidth=1.5,
                linestyle=line_styles[i], label=label)
    winds = wind_range_from_label(legend[0])[0]
    title = 'Distribution of the normalized Horns Rev wind farm power ouput'
    ax = ax = general_axes_property(ax, 'Wind Direction(degree)', 'Normalized Power',
                                    (winds[0] - (len(winds) / 10), winds[-1] + (len(winds) / 10)),
                                    (0.35, 1.05), len(winds) // 10, title)
    if kwargs.get("psave", False):
        plt.savefig("output/{}.png".format(kwargs["psave"]), format='png',
                    dpi=300, bbox_inches='tight')
        print("** Picture {} Save Done ! **".format(kwargs["psave"]))
    if kwargs.get("dsave", False):
        pd.DataFrame(data.T, columns=legend).to_csv(
            "output/{}.csv".format(kwargs["dsave"]))
        print("**  Data {} Save Done ! **".format(kwargs["dsave"]))
    plt.show()


def custom_power_eval(data, ref_data, psave, pshow=None):
    if isinstance(data, str):
        data = pd.read_csv("{}/output/{}.csv".format(root, data))
        print("** Data {} Loaded Done !** \n".format(data))

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    colors = list(mcolors.TABLEAU_COLORS.keys())

    x = np.arange(1, data.shape[1] + 1)
    for i, tag in enumerate(ref_data.columns):
        ax.plot(x, ref_data[tag].values,
                linestyle=line_styles[i],
                c=colors[i],
                lw=0.00,
                label=tag,
                markersize=6,
                marker=marker_styles[i],
                markeredgecolor='k',
                markeredgewidth=2)
    for i in range(data.shape[0]):
        ax.plot(x, power_ops.normalized_wt_power(data[i, :]),
                color=colors[i], linewidth=2, linestyle=line_styles[i],
                markersize=5, marker=marker_styles[-(i + 1)],
                label="")
    title = ""
    ax = general_axes_property(ax, 'Turbine Row', 'Normalized Power',
                               (0.5, data.shape[1] + 0.5), (0.1, 1.05),
                               1, title)
    if psave:
        plt.savefig("{}/output/{}.png".format(root, psave), format='png',
                    dpi=300, bbox_inches='tight')
        print("** Picture {} Save Done !** \n".format(psave))
    if pshow:
        plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def wind_range_from_label(label):
    s, e = re.search(r"(\([^\)]+\))", label).span()
    w = label[s: e + 1] if s == 0 else label[s - 1: e]
    label = label.replace(w, "")
    wind_range = eval(w.strip("+"))
    return np.arange(wind_range[0], wind_range[1]), label


def power_plot_saved_data(fname, extract=None):
    if not isinstance(fname, list):
         return pd.read_csv(fname)[extract] if extract is not None else pd.read_csv(fname)
    data = pd.concat([
        pd.read_csv(f) for f in map(lambda x: f"output/{x}", fname)], axis=1)
    if extract is None or not (isinstance(extract, list)):
         return pd.read_csv(fname)[extract] if extract is not None else pd.read_csv(fname)
    return data[extract]


def general_axes_property(axes, *args):
    axes.set_xlabel(args[0], font1)
    axes.set_ylabel(args[1], font1)
    axes.set_xlim(args[2])
    axes.set_ylim(args[3])
    axes.tick_params(labelsize=10)
    labels = axes.get_xticklabels() + axes.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    axes.xaxis.set_major_locator(MultipleLocator(args[4]))
    rcParams['xtick.direction'], rcParams['ytick.direction'] = 'in', 'in'
    axes.legend(loc="best", prop=font2, edgecolor='None', frameon=False,
                labelspacing=0.4, bbox_transform=axes.transAxes)
    axes.set_title(args[5], fontdict=font2)
    return axes


def powers_table_format(powers, cost):
    table = np.zeros((powers.shape[0] + 1, 3))
    table[0, 1], table[0, 2] = powers[0, 1], 0
    powers[:, 1] = ((powers[:, 1] - powers[:, 0]) / powers[:, 1]) * 100
    table[:, 0], table[1:, 1:] = np.arange(powers.shape[0] + 1), powers
    table[:, 1], table[0, 2], table[1:, 2] = \
        np.around(table[:, 1], decimals=3), np.around(table[0, 2]), \
            np.around(table[1:, 2], decimals=2)
    table = [[int(table[i, 0])] + list(table[i, 1:]) for i in range(table.shape[0])]
    power, no_wake_power = np.sum(powers, axis=0)[0], np.sum(powers, axis=0)[1]
    loss = (no_wake_power - power) * 100 / no_wake_power
    table = list(table) + [["Total", round(power, 3), round(loss, 2)]] + \
        [["LCOE", round(cost, 2), "€/MWh"]]
    # print(table)
    return table


# if __name__ == "__main__":
#     pass