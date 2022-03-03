import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

from floris.utils.visualization import property as ppt


opt_lcoe_25 = [76.90, 85.07, 80.87]
opt_lcoe_36 = [77.21, 86.11, 81.38]
opt_lcoe_49 = [79.32, 87.01, 82.25]
opt_lcoe = [opt_lcoe_25, opt_lcoe_36, opt_lcoe_49]

ba_lcoe_25 = [77.47, 85.45, 81.52]
ba_lcoe_36 = [78.02, 86.73, 82.62]
ba_lcoe_49 = [80.73, 88.15, 83.86]
ba_lcoe = [ba_lcoe_25, ba_lcoe_36, ba_lcoe_49]

opt_aep_25 = [28.50, 26.54, 26.82]
opt_aep_36 = [40.66, 36.76, 38.31]
opt_aep_49 = [54.32, 48.92, 51.68]
opt_aep = [opt_aep_25, opt_aep_36, opt_aep_49]

ba_aep_25 = [28.10, 25.27, 26.59]
ba_aep_36 = [40.15, 35.81, 37.74]
ba_aep_49 = [53.12, 47.89, 50.54]
ba_aep = [ba_aep_25, ba_aep_36, ba_aep_49]

opt_cf_25 = [56.99, 51.24, 53.64]
opt_cf_36 = [56.47, 50.46, 53.02]
opt_cf_49 = [55.43, 50.12, 52.72]
opt_cf = [opt_cf_25, opt_cf_36, opt_cf_49]

ba_cf_25 = [56.20, 50.55, 53.18]
ba_cf_36 = [55.76, 49.74, 52.41]
ba_cf_49 = [54.21, 48.87, 51.57]
ba_cf = [ba_cf_25, ba_cf_36, ba_cf_49]

opt_eff_25 = [99.15, 97.21, 97.84]
opt_eff_36 = [99.23, 96.42, 96.12]
opt_eff_49 = [97.64, 94.65, 95.25]
opt_eff = [opt_eff_25, opt_eff_36, opt_eff_49]

ba_eff_25 = [98.92, 96.63, 97.01]
ba_eff_36 = [98.14, 95.08, 95.60]
ba_eff_49 = [96.32, 93.43, 94.07]
ba_eff = [ba_eff_25, ba_eff_36, ba_eff_49]

opt = [opt_lcoe, opt_aep, opt_cf, opt_eff]
ba = [ba_lcoe, ba_aep, ba_cf, ba_eff]


def result_comp_bar_1(opt, ba):
    n = 3
    bar_width = 0.12
    x = np.array([1. + i * 4 * bar_width for i in range(n)])
    tick_labels = ["W-S", "W-FA", "W-FV"]
    patterns = ('', 'x', '//')
    colors = ['b', 'g', 'r']
    ylims = [[75., 90.], [23., 55.], [48., 58.], [93., 100.]]
    ylabels = ["LCOE (€/MWh)",
            "AEP (MW)",
            "Capacity factor (%)",
            "Efficiency (%)"]
    labels = [r"$\mathit{N_t}=25$",
            r"$\mathit{N_t}=36$",
            r"$\mathit{N_t}=49$"]
    numbers = ["(a)", "(b)", "(c)", "(d)"]
    text_yaxis = [89.0, 53.0, 57.5, 99.6]

    # fig = plt.figure(figsize=(6, 8), dpi=120, facecolor="white")
    # axes = plt.subplot(111)
    fig, axarr = plt.subplots(2, 2, figsize=(14, 16),
                            dpi=120, facecolor="white")
    axarr = axarr.flatten()

    for j, ax in enumerate(axarr):
        for i, (optv, bav) in enumerate(zip(opt[j], ba[j])):
            a1 = 1.0 if j == 0 else 0.5
            v1, v2 = (optv, bav) if j == 0 else (bav, optv)
            c1, c2 = (colors[i], 'w') if j == 0 else ('k', colors[i])
            h1, h2 = (patterns[0], patterns[2]) if j == 0 else (patterns[2], patterns[0])
            offset = np.array(v2) - np.array(v1)

            ax.bar(x + i * bar_width, v1,
                bar_width,
                color=c1,
                align='center',
                label=labels[i],
                hatch=h1,
                linewidth=1.0,
                edgecolor='k',
                alpha=a1,
                )
            ax.bar(x + i * bar_width, offset,
                bar_width,
                bottom=v1,
                color=c2,
                align='center',
                #    label=labels[i],
                hatch=h2,
                linewidth=1.0,
                edgecolor='k',
                alpha=1.0,
                )

            # axes.bar(x + i * 2 * bar_width, opt,
            #         bar_width,
            #         color=colors[i],
            #         align='center',
            #         # label='Optimized',
            #         hatch=patterns[0],
            #         linewidth=1.0,
            #         edgecolor='k',
            #         alpha=aa,
            #         )

            # axes.bar(x + (i * 2 + 1) * bar_width, ba,
            #         bar_width,
            #         color=colors[i],
            #         align='center',
            #         # label='Baseline',
            #         hatch=patterns[1],
            #         linewidth=1.0,
            #         edgecolor='k',
            #         alpha=aa,
            #         )
        ticks = ax.get_xticklabels() + ax.get_yticklabels()
        [tick.set_fontname('Times New Roman') for tick in ticks]
        ax.tick_params(labelsize=15, colors='k', direction='in',
                    top=True, bottom=True, left=True, right=True)
        ax.set_xlabel(r'Wind scenario', ppt.font15)
        ax.set_xticks(x + (n - 1) * bar_width / 2)
        ax.set_xticklabels(tick_labels)
        ax.set_xticklabels([])
        ax.set_ylim(ylims[j])
        ax.set_ylabel(ylabels[j], ppt.font15)
        ax.text(2.2, text_yaxis[j], f"${numbers[j]}$", fontsize=15, color='k')

    fig.subplots_adjust(hspace=0.2)
    axarr[0].legend(loc='upper left', prop=ppt.font15, edgecolor='None', frameon=False,
                    labelspacing=0.4, bbox_transform=axarr[0].transAxes)

    # plt.savefig('./result_comp.png', format='png', dpi=300, bbox_inches='tight', )
    plt.show()


def result_comp_bar_2(opt, ba):
    opt = np.array(opt).transpose(0, 2, 1)
    ba = np.array(ba).transpose(0, 2, 1)

    n = 3
    bar_width = 0.12
    x = np.array([1. + i * 4 * bar_width for i in range(n)])
    labels = ["W-S", "W-FA", "W-FV"]
    sub_labels = ["Reduction by optimization", "Improvement by optimization"]
    patterns = ('', 'x', '//')
    colors = ['b', 'g', 'r']
    ylims = [[75., 90.], [23., 55.], [48., 60.5], [93., 101.5]]
    ylabels = ["LCOE (€/MWh)",
            "AEP (MW)",
            "Capacity factor (%)",
            "Efficiency (%)"]
    tick_labels = [r"$\mathit{N_t}=25$",
                   r"$\mathit{N_t}=36$",
                   r"$\mathit{N_t}=49$"]
    numbers = ["(a)", "(b)", "(c)", "(d)"]
    text_yaxis = [89.0, 53.0, 59.7, 100.9]

    # fig = plt.figure(figsize=(6, 8), dpi=120, facecolor="white")
    # axes = plt.subplot(111)
    fig, axarr = plt.subplots(2, 2, figsize=(15, 12),
                              dpi=120, facecolor="white")
    axarr = axarr.flatten()

    for j, ax in enumerate(axarr):
        rects1, rects2 = [], []
        for i, (optv, bav) in enumerate(zip(opt[j], ba[j])):
            a1 = 1. if j == 0 else 1.
            a2 = .5 if j == 0 else 1.
            v1, v2 = (optv, bav) if j == 0 else (bav, optv)
            c1, c2 = (colors[i], 'k') if j == 0 else (colors[i], 'w')
            h1, h2 = (patterns[0], patterns[2]) if j == 0 else (patterns[0], patterns[2])
            offset = np.array(v2) - np.array(v1)
            l1, l2 = (labels[i], 'Reduction by optimization') if j == 0 else \
                (labels[i], 'Improvement by optimization')

            r1 = ax.bar(x + i * bar_width, v1,
                            bar_width,
                            color=c1,
                            align='center',
                            # label=l1,
                            hatch=h1,
                            linewidth=1.0,
                            edgecolor='k',
                            alpha=a1,
                            )
            r2 = ax.bar(x + i * bar_width, offset,
                            bar_width,
                            bottom=v1,
                            color=c2,
                            align='center',
                            # label=l2,
                            hatch=h2,
                            linewidth=1.0,
                            edgecolor='k',
                            alpha=a2,
                            )
            rects1.append(r1[0])
            rects2.append(r2[0])

            # axes.bar(x + i * 2 * bar_width, opt,
            #         bar_width,
            #         color=colors[i],
            #         align='center',
            #         # label='Optimized',
            #         hatch=patterns[0],
            #         linewidth=1.0,
            #         edgecolor='k',
            #         alpha=aa,
            #         )

            # axes.bar(x + (i * 2 + 1) * bar_width, ba,
            #         bar_width,
            #         color=colors[i],
            #         align='center',
            #         # label='Baseline',
            #         hatch=patterns[1],
            #         linewidth=1.0,
            #         edgecolor='k',
            #         alpha=aa,
            #         )
        ax.set_xlabel(r'Wind turbine number', ppt.font15)
        ax.set_xticks(x + (n - 1) * bar_width / 2)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(ylims[j])
        ax.set_ylabel(ylabels[j], ppt.font15)
        ticks = ax.get_xticklabels() + ax.get_yticklabels()
        [tick.set_fontname('Times New Roman') for tick in ticks]
        ax.tick_params(labelsize=15, colors='k', direction='in',
                    top=True, bottom=True, left=True, right=True)
        ax.text(2.2, text_yaxis[j], f"${numbers[j]}$", fontsize=15, color='k')
        ax.legend(rects1, labels, loc='upper left', prop=ppt.font15, edgecolor='None',
                  frameon=False, labelspacing=0.4, columnspacing=0.6, bbox_transform=ax.transAxes,
                  ncol=3, handletextpad=0.15)
        lab = sub_labels[0] if j == 0 else sub_labels[1]
        leg = Legend(ax, (rects2[0],), (lab,), loc='upper left', prop=ppt.font15,
                     edgecolor='None', frameon=False, labelspacing=0.4,
                     columnspacing=0.5, bbox_transform=ax.transAxes,
                     bbox_to_anchor=(0.0, 0.92),)
        ax.add_artist(leg)

    fig.subplots_adjust(hspace=0.2)
    plt.savefig('./result_comp.png', format='png', dpi=300, bbox_inches='tight', )
    plt.show()


if __name__ == "__main__":
    result_comp_bar_2(opt, ba)

    # print(np.array(opt).transpose(0, 2, 1))
