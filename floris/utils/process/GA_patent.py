from tkinter import W
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from floris.utils.visual import property as ppt



def optimization_boxplot():
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    np.random.seed(1236)
    means = np.array([81.81, 81.39 + 0.6, 81.38])
    covariance = np.array([[0.15, 0., 0.],[0., 0.1, 0.],[0., 0., 0.05]])
    gaussian = np.random.multivariate_normal(means, covariance, (10, ))
    print(gaussian)
    # label = ['遗传', '粒子群', '融合']
    label = ['GA', 'PSO', 'Hybrid']
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
    ax.set_ylabel('LCOE(€/MWh)',
                  {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 20, })
    ax.set_ylim([80., 83.])
    # ax.set_yticks(0.5 * np.arange(5))
    # ax.set_yticklabels(['0', '', '1', '', '2'])
    # ax.xaxis.label.set_size(20)
    # ax.yaxis.label.set_size(18)
    ax.tick_params(axis='x', colors='k', direction='in', which='both',
                   width=1., top=True, bottom=True, labelsize=20)
    ax.tick_params(axis='y', colors='k', direction='in', which='both',
                   width=1., left=True, right=True, labelsize=17)
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]

    plt.show()


def optimization_layout(layout=None, baseline=None, **kwargs):
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    dist = kwargs.get("normal", 80.)
    legend_loc = 'upper left' if kwargs.get("opt_data", False) else 'upper right'
    if layout is not None:
        opt_label = "Optimized WTs: LCOE = XX €/MWh, Power = XX MW, CF = XX %" \
            if kwargs.get("opt_data", False) else "Optimized WTs"
        ax.plot(layout[:, 0] / dist,
                layout[:, 1] / dist,
                linestyle="-",
                c="w",
                lw=0.00,
                zorder=1,
                label=opt_label,
                markersize=6,
                marker="o",
                markeredgecolor='r',
                markeredgewidth=1.2)
        if kwargs.get("annotate", False):
            x, y = layout[:, 0], layout[:, 1]
            num_labs = [str(i) for i in range(1, len(x) + 1)]
            for i in range(len(num_labs)):
                plt.annotate(num_labs[i], xy=(x[i], y[i]),
                            xytext=(x[i] + 50, y[i] + 50))
    if baseline is not None:
        # marker = 'o' if layout is None else 'x'
        # alpha = 1. if layout is None else 0.6
        ref_label = "Baseline WTs: LCOE = XX €/MWh, Power = XX MW, CF = XX %" \
            if kwargs.get("ref_data", False) else "Baseline WTs"
        ax.plot(baseline[:, 0] / dist,
                baseline[:, 1] / dist,
                linestyle="-",
                c="k",
                lw=0.00,
                alpha=0.6,
                label=ref_label,
                markersize=6,
                marker='x',
                markeredgecolor='k',
                markeredgewidth=1.5)

    if (layout is not None) or (baseline is not None):
        ax.set_xlim((-6 * 80. / dist, 70 * 80. / dist))
        ax.set_ylim((-6 * 80. / dist, 70 * 80. / dist))
        ax.set_xlabel(r'Normalized distance($\mathit{D}$)', ppt.font13)
        ax.set_ylabel(r'Normalized distance($\mathit{D}$)', ppt.font13)
        # ax.xaxis.set_ticks_position('top')
        # ax.yaxis.set_ticks_position('right')
        ax.tick_params(labelsize=12, colors='k', direction='in',
                    top=True, bottom=True, left=True, right=True)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        default_bounds = np.array([[0, 5040, 5040, 0], [3911, 3911, 0, 0]])
        bounds = kwargs.get("bounds", default_bounds)
        bounds_label = "WF boundary" if kwargs.get("bounds_label", False) else ''
        xs, ys = bounds[0, :] / dist, bounds[1, :] / dist
        patch = patches.Polygon(xy=list(zip(xs, ys)), fill=False, linewidth=1,
                                edgecolor='b', facecolor='none', linestyle="--",
                                alpha=0.8, label=bounds_label)
        ax.add_patch(patch)

        ax.legend(loc=legend_loc, prop=ppt.font13, edgecolor='None', frameon=False,
                labelspacing=0.4, bbox_transform=ax.transAxes,)

        fname = kwargs.get("layout_name", "layout")
        path = kwargs.get("path", "solution")
        fpath = f"{path}/{fname}.png"
        plt.savefig(fpath, format='png', dpi=300, bbox_inches='tight')
        print(f"Optimized/Baseline Layout Save Done ({fpath}).")
        plt.show()
    else:
        print("No Layout to be plotted.")



if __name__ == "__main__":
    optimization_boxplot()