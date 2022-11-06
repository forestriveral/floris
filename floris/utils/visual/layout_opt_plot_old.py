import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator

from floris.utils.visual import plot_property as ppt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                  OPTIMIZATION                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def wf_layout_show_old(layout, ref=None, show=True, save=True,
                       spath=None, annotate=False):
    x, y = layout[:, 0], layout[:, 1]
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(x, y,
            linestyle="-",
            c="g",
            lw=0.00,
            label="Optimal",
            markersize=6,
            marker="o",
            markeredgecolor='k',
            markeredgewidth=1)
    if ref is not None:
        ax.plot(ref[:, 0], ref[:, 1],
                linestyle="-",
                c="k",
                lw=0.00,
                alpha=0.6,
                label="Horns",
                markersize=8,
                marker="x",
                markeredgecolor=None,
                markeredgewidth=2)
    # plt.xlim((-8 * 80., 80 * 80.));plt.ylim((-4 * 80., 70 * 80.))
    if annotate:
        num_labs = [str(i) for i in range(1, len(x) + 1)]
        for i in range(len(num_labs)):
            plt.annotate(num_labs[i], xy=(x[i], y[i]),
                         xytext=(x[i] + 50, y[i] + 50))
    if save:
        save_path = f"output/{spath}/layout.png" if spath else "solution/layout.png"
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"** Layout Picture Save Done ({save_path})!** ")
    plt.legend(loc="upper right")
    if show:
        print('Optimal wind farm layout plotted done.')
        plt.show()


def wf_layout_plot(layout=None, baseline=None, **kwargs):
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


def wd_power_plot(wds, powers, capacity, **kwargs):
    cf, eff =  powers[:, 0] / np.sum(capacity), powers[:, 0] / powers[:, 1]
    powers = np.concatenate((powers, cf[:, None], eff[:, None]), axis=1)

    fig = plt.figure(figsize=(10, 8), dpi=150)
    axl = fig.add_subplot(111)
    plt.rc('font',family='Times New Roman')
    labels = ["Power(MW)", "No-wake Power(MW)", 
              "CF(%)", "Eff(%)", "Loss(%)"]
    colors = ['k', 'k', 'b', 'b']
    styles = ['-', '--', '-', '--']

    lines = []
    for i in range(2):
        ln = axl.plot(wds, powers[:, i],
                      color=colors[i],
                      label=labels[i],
                      linewidth=2,
                      linestyle=styles[i],
                      )
        lines.append(ln)
    axl.set_xlim([-10., 370.])
    axl.set_xlabel(r'Wind Directions $\mathit{\theta}(^o)$', ppt.font15)
    axl.set_xticks(np.arange(0, 361, 60))
    # axl.set_xticklabels(['5', '10', '15', '20',])
    # axl.set_ylim([0, 2.10])
    axl.set_ylabel('Wind Farm Power (MW)', ppt.font15)
    # axl.set_yticks([0, 0.5, 1, 1.5, 2,])
    # axl.set_yticklabels(['0', '0.5', '1', '1.5', '2'])
    axl.tick_params(labelsize=15, direction='in')
    axl.grid(linestyle=':', linewidth=0.5, color='k', alpha=0.8, zorder=0)

    axr = axl.twinx()
    for i in range(2):
        ln = axr.plot(wds, powers[:, i + 2],
                      color=colors[i + 2],
                      label=labels[i + 2],
                      linewidth=2,
                      linestyle=styles[i + 2],
                      )
        lines.append(ln)
    axr.set_ylim([-0.1, 1.1])
    axr.set_ylabel('Capacity Factor/Efficiency (%)', ppt.font15b)
    axr.set_yticks([0, 0.25, 0.5, 0.75, 1.,])
    axr.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
    axr.tick_params(labelsize=15, colors='b', direction='in')
    axr.spines['right'].set_color('b')

    # added these lines
    lines = lines[0] + lines[1]+ lines[2] + lines[3]
    labs = [l.get_label() for l in lines]
    axl.legend(lines, labs, loc='lower center', prop=ppt.font15,
               edgecolor='None', frameon=False, labelspacing=0.4,
               bbox_transform=axl.transAxes,
            #    bbox_to_anchor=(1.0, 0.7),
               )

    labels = axl.get_xticklabels() + axl.get_yticklabels() + axr.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    fname = kwargs.get("wd_name", "wds_powers")
    path = kwargs.get("path", "solution")
    fpath = f"{path}/{fname}.png"
    plt.savefig(fpath, format='png', dpi=300, bbox_inches='tight')
    print(f'WDs Power Data of Optimal Layout Save Done ({fpath})')
    plt.show()


def wt_power_plot(powers, capacity, **kwargs):
    cf, eff = powers[:, 0] * 100 / capacity, powers[:, 0] * 100 / powers[:, 1]
    powers = np.concatenate(
        (powers, cf[:, None], eff[:, None], 100. - eff[:, None]), axis=1)
    cols = ["Power[MW]", "No-wake[MW]", "CF[%]", "Eff[%]", "Loss[%]"]
    powers_data = pd.DataFrame(np.around(powers.astype(float), decimals=3), columns=cols,
                               index=np.arange(1, powers.shape[0] + 1))
    fname = kwargs.get("wt_name", "wts_powers")
    path = kwargs.get("path", "solution")
    pd_fpath = f"{path}/{fname}.csv"
    powers_data.to_csv(pd_fpath, )
    print(f'WTs Power Data of Optimal Layout Save Done ({pd_fpath})')

    fig = plt.figure(figsize=(10, 8), dpi=150)
    axl = fig.add_subplot(111)
    plt.rc('font',family='Times New Roman')
    labels = ["Power(MW)", "No-wake Power(MW)", 
              "CF(%)", "Eff(%)", "Loss(%)"]
    colors = ['k', 'k', 'b', 'b']
    styles = ['-', '--', '-', '--']
    lines = []
    wts = np.arange(1, powers.shape[0] + 1)
    width = 0.35
    rects1 = axl.bar(wts - width / 2,
                     powers[:, 2],
                     color='b',
                     label='Power(MW)',
                     linewidth=0.5,
                     linestyle='-',
                     )
    # axl.set_xlim([-10., 370.])
    # axl.set_xlabel(r'Wind Directions $\mathit{\theta}(^o)$', ppt.font15)
    # axl.set_xticks(np.arange(0, 361, 60))
    # axl.set_xticklabels(['5', '10', '15', '20',])
    axl.set_ylim([np.min(powers[:, 2]) * 0.8, np.max(powers[:, 2]) * 1.2])
    axl.set_ylabel('Wind Turbine Power (MW)', ppt.font15)
    # axl.set_yticks([0, 0.5, 1, 1.5, 2,])
    # axl.set_yticklabels(['0', '0.5', '1', '1.5', '2'])
    axl.tick_params(labelsize=15, direction='in')
    # axl.grid(linestyle=':', linewidth=0.5, color='k', alpha=0.8, zorder=0)

    labels = axl.get_xticklabels() + axl.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    pic_fpath = f"{path}/{fname}.png"
    plt.savefig(pic_fpath, format='png', dpi=300, bbox_inches='tight')
    plt.show()


def opt_curve_plot(results, **kwargs):
    if results['config']['stage'] == 2:
        fbests = [results['fbest'][0] + results['fbest'][1], ]
        favgs = [results['favg'][0] + results['favg'][1], ]
        labels = ['ga-pso fbest', 'ga-pso favg']
    else:
        fbests, favgs = [results['fbest'], ], [results['favg'], ]
        labels = ['fbest', 'favg']

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)
    plt.rc('font',family='Times New Roman')
    # labels = []
    colors = ['k', 'b', 'r', 'g']
    # styles = ['-', '-', ':', '-']
    if fbests:
        # fbests curve plot
        fbests = np.array(fbests)
        for i in range(fbests.shape[0]):
            ax.plot(np.arange(fbests.shape[1]),
                    fbests[i, :],
                    color=colors[i],
                    label=labels[i],
                    linewidth=2,
                    linestyle=':',
                    )
    if favgs:
        # favgs curve plot
        favgs = np.array(favgs)
        for i in range(favgs.shape[0]):
            ax.plot(np.arange(favgs.shape[1]),
                    favgs[i, :],
                    color=colors[i],
                    label=labels[i + fbests.shape[0]],
                    linewidth=2,
                    linestyle='-',
                    )
    ax.set_xlim([0., favgs.shape[1]])
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel('Generation', ppt.font15)
    # ax.set_xticks(np.arange(0, 361, 60))
    # axl.set_xticklabels(['5', '10', '15', '20',])
    # ax.set_ylim([0, 2.10])
    # ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel('LCOE (€/MWh)', ppt.font15)
    # axl.set_yticks([0, 0.5, 1, 1.5, 2,])
    # axl.set_yticklabels(['0', '0.5', '1', '1.5', '2'])
    ax.tick_params(labelsize=15, direction='in')
    # ax.grid(linestyle=':', linewidth=0.5, color='k', alpha=0.8, zorder=0)

    ax.legend(loc='best', prop=ppt.font15, edgecolor='None', frameon=False,
              labelspacing=0.4, bbox_transform=ax.transAxes)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    fname = kwargs.get("curve_name", "opt_curve")
    path = kwargs.get("path", "solution")
    fpath = f"{path}/{fname}.png"
    plt.savefig(fpath, format='png', dpi=300, bbox_inches='tight')
    print(f'Optimization Curve Picture Save Done ({fpath})')
    plt.show()


def table_legend_plot(table, show=True, save=True, spath=None):
    fig = plt.figure()
    fig.subplots_adjust(top=0.1, bottom=0.05, left=0.1, wspace=0.5)
    table_subplot = plt.subplot2grid((1, 4), (0, 3))
    table = plt.table(
        cellText=table,
        colWidths=[0.05]*3,
        colLabels=['WT', 'P[MW]', 'Loss[%]'],
        colColours=['#F3CC32', '#2769BD', '#DC3735'],
        # rowColours=['#FFFFFF' for i in range(len(table) - 1)] + ['#228B22'],
        loc='center',
        cellLoc='center',)
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width((-1, 0, 1, 2, 3))
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    for (row, col), cell in table.get_celld().items():
        if (row == 0):
            cell.set_text_props(
                fontproperties=FontProperties(weight='bold', size=7))
    plt.axis('off')
    if save:
        save_path = f"output/{spath}/powers_table.png" if spath else \
            "solution/powers_table.png"
        plt.savefig(save_path, format='png', dpi=300,
                    bbox_inches='tight', )
        print(f"** Power Table Legend Save Done ({save_path})! ** ")
    if show:
        plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #






# if __name__ == "__main__":
#     pass

