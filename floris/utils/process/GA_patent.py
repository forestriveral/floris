from tkinter import W
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from matplotlib.font_manager import FontProperties

from floris.utils.visual import plot_property as ppt
from floris.utils.tools import horns_farm_config as horns_config



def optimization_boxplot():
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.size'] = '20'
    plt.rcParams['font.sans-serif'] = ['SimSun']
    # plt.rcParams['font.sans-serif'] = ['Minion-Pro', 'SimSun']
    # plt.rcParams['axes.unicode_minus'] = False

    # config = {
    #     "font.family":'serif',
    #     "font.serif": ['SimSun'],
    #     "font.size": 10,
    #     "axes.unicode_minus": False
    #     "mathtext.fontset":'stix',
    #     }
    # plt.rcParams.update(config)
    # SimSun = FontProperties(fname='filepath/TimesSong.ttf')

    np.random.seed(1236)
    means = np.array([81.81 + 0.3, 81.39 + 0.6 + 0.2, 81.38 - 0.3])
    covariance = np.array([[0.15, 0., 0.],[0., 0.1, 0.],[0., 0., 0.05]])
    gaussian = np.random.multivariate_normal(means, covariance, (10, ))
    print(gaussian)
    print('Min', gaussian.min(0))
    print('Avg', gaussian.mean(0))
    print('Max', gaussian.max(0))
    print('Up', (gaussian.min(0)[-1] - gaussian.min(0)[0]) / gaussian.min(0)[0] * 100.,
          (gaussian.min(0)[-1] - gaussian.min(0)[1]) / gaussian.min(0)[1] * 100.)
    label = ['遗传', '粒子群', '融合']
    ylabel = '平准化能源成本(€/MWh)'
    # label = ['GA', 'PSO', 'Hybrid']
    # ylabel = 'LCOE(€/MWh)'
    color = ['k', 'b', 'r']

    _, ax = plt.subplots(figsize=(8, 6), dpi=120)
    boxes = plt.boxplot(gaussian,
                        sym='x',
                        labels=label,
                        widths=0.5,
                        showmeans=True,
                        showfliers=True,
                        )
    for box, median, flier, mean, c in zip(
        boxes['boxes'], boxes['medians'], boxes['fliers'], boxes['means'], color):
        box.set(color=c, linewidth=2)
        # box.set(boxesacecolor=c)
        median.set(color=c, linewidth=2)
        flier.set(marker='x', color=c, alpha=1.)
        mean.set(marker='s', markerfacecolor='w',
                 markeredgecolor=c, markersize=9,
                 markeredgewidth=0.8)
    for i, (whisker, cap) in enumerate(zip(boxes['whiskers'], boxes['caps'])):
        whisker.set(color=color[i // 2], linewidth=2)
        cap.set(color=color[i // 2], linewidth=2)

    # ax.set_xlabel('Optimization method',
    #               {'family': 'Times New Roman',
    #                'weight': 'normal',
    #                'size': 20, })
    # ax.set_xlim([0., 12.])
    # ax.set_xticks(np.arange(0, 13, 1))
    # ax.set_xticks(np.arange(0, 12, 0.5), minor=True)
    # ax.set_xticklabels([str(i) for i in np.arange(13)])
    # ax.set_xticklabels([str(int(i)) if int(i) == i else '' for i in 0.5 * np.arange(23)])
    ax.set_ylabel(ylabel, {'family': 'sans-serif', 'weight': 'normal', 'size': 20, })
    ax.set_ylim([80., 83.])
    # ax.set_yticks(0.5 * np.arange(5))
    # ax.set_yticklabels(['0', '', '1', '', '2'])
    # ax.xaxis.label.set_size(20)
    # ax.yaxis.label.set_size(18)
    ax.tick_params(axis='x', colors='k', direction='in', which='both',
                   width=1., top=True, bottom=True, labelsize=20)
    ax.tick_params(axis='y', colors='k', direction='in', which='both',
                   width=1., left=True, right=True, labelsize=17)
    # [xticklab.set_fontname('Times New Roman') for xticklab in ax.get_xticklabels()]
    [yticklab.set_fontname('Times New Roman') for yticklab in ax.get_yticklabels()]
    # plt.savefig('../outputs/comparison_patent.png', format='png', dpi=200, bbox_inches='tight')

    plt.show()


def optimization_layout(method=None, num=36):  # sourcery skip: move-assign
    plt.rcParams['font.sans-serif'] = ['SimSun']
    baseline = horns_config.Horns.baseline(num)
    settings = {'GA': [1234, [0., 1.]],
                'PSO': [1235, [0., 0.8]],
                'Hybrid': [2225, [0., 0.9]]}
    bounds = np.array([[0, 5040, 5040, 0], [3911, 3911, 0, 0]])
    offset = layout_offset_generator(settings.get(method, None))
    # print(offset)
    # optimized_label = 'Optimized turbines'
    # baseline_label = 'Baseline turbines'
    # boundary_label = 'Layout boundary'
    # xyaxis_label =  'Normalized Distance'
    optimized_label = '优化风机位置'
    baseline_label = '标准风机位置'
    boundary_label = '布局优化边界'
    xyaxis_label =  '归一化距离(轮径)'

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    D_rotor = 80.
    if offset is not None:
        layout = baseline / D_rotor + offset * 2.
        ax.plot(np.clip(layout[:, 0], 0, 5040 / D_rotor),
                np.clip(layout[:, 1], 0, 3911 / D_rotor),
                linestyle="-",
                c="w",
                lw=0.00,
                zorder=1,
                label=optimized_label,
                markersize=6,
                marker="o",
                markeredgecolor='r',
                markeredgewidth=1.2)
    ax.plot(baseline[:, 0] / D_rotor,
            baseline[:, 1] / D_rotor,
            linestyle="-",
            c="k",
            lw=0.00,
            alpha=0.6,
            label=baseline_label,
            markersize=6,
            marker='x',
            markeredgecolor='k',
            markeredgewidth=1.5)

    ax.set_xlim((-6 * 80. / D_rotor, 70 * 80. / D_rotor))
    ax.set_ylim((-6 * 80. / D_rotor, 70 * 80. / D_rotor))
    ax.set_xlabel(xyaxis_label, {'family': 'sans-serif','weight': 'normal', 'size': 20,})
    ax.set_ylabel(xyaxis_label, {'family': 'sans-serif','weight': 'normal', 'size': 20,})
    # ax.xaxis.set_ticks_position('top')
    # ax.yaxis.set_ticks_position('right')
    ax.tick_params(labelsize=18, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    [xticklab.set_fontname('Times New Roman') for xticklab in ax.get_xticklabels()]
    [yticklab.set_fontname('Times New Roman') for yticklab in ax.get_yticklabels()]

    xs, ys = bounds[0, :] / D_rotor, bounds[1, :] / D_rotor
    ax.add_patch(patches.Polygon(xy=list(zip(xs, ys)), fill=False, linewidth=1,
                                    edgecolor='b', facecolor='none', linestyle="--",
                                    alpha=0.8, label=boundary_label))
    ax.legend(loc='upper left', edgecolor='None', frameon=False, labelspacing=0.4,
              bbox_transform=ax.transAxes, prop={'family': 'sans-serif','weight': 'bold', 'size': 15})

    plt.savefig(f'../outputs/{method}_patent.png', format='png', dpi=200, bbox_inches='tight')
    plt.show()


def layout_offset_generator(setting):
    if setting is None:
        return None
    print(setting[0])
    np.random.seed(setting[0])
    return np.random.normal(loc=setting[1][0], scale=setting[1][1], size=(36, 2))



if __name__ == "__main__":
    optimization_boxplot()
    # optimization_layout('GA')
    # optimization_layout('PSO')
    # optimization_layout('Hybrid')