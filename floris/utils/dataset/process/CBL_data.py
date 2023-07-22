import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from floris.utils.module.tools import plot_property as ppt


data_dir = "../data/others/CBL_simulation/data"


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                            ISHIHARA_DATA_PLOT                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def cbl_vertical_velocity_profile(param='vel'):
    distance_list = [0, 2, 3, 5, 7, 9, 10]
    vertical_vel_exp, vertical_vel_rsm = [], []
    for dist in distance_list:
        exp_data = f'{data_dir}/v_{param}_{dist}d_exp.txt'
        rsm_data = f'{data_dir}/v_{param}_{dist}d_rsm.txt'
        if os.path.exists(exp_data):
            vertical_vel_exp.append(np.loadtxt(exp_data, skiprows=4))
        else:
            vertical_vel_exp.append([])
        if os.path.exists(rsm_data):
            vertical_vel_rsm.append(np.loadtxt(rsm_data, skiprows=4))
        else:
            vertical_vel_rsm.append([])

    fig, ax = plt.subplots(figsize=(12, 4), dpi=120)
    ax.set_xlabel('x/D', ppt.font18t)
    ax.set_xlim([0., 12.])
    ax.set_xticks(np.arange(0, 13, 1))
    ax.set_xticks(np.arange(0, 12, 0.5), minor=True)
    ax.set_xticklabels([str(i) for i in np.arange(13)])
    # ax.set_xticklabels([str(int(i)) if int(i) == i else '' for i in 0.5 * np.arange(23)])
    ax.set_ylabel('z/H', ppt.font18t)
    ax.set_ylim([0., 2.])
    ax.set_yticks(0.5 * np.arange(5))
    ax.set_yticklabels(['0', '', '1', '', '2'])

    for i, dist in enumerate(distance_list):
        if np.any(vertical_vel_rsm[i]):
            rsm_x, rsm_y = vertical_vel_rsm[i][:, 0], vertical_vel_rsm[i][:, 1]
            rsm_x = rsm_x + dist if param == 'vel' else rsm_x / 0.5 + dist
        else:
            rsm_x, rsm_y = [], []

        if np.any(vertical_vel_exp[i]):
            exp_x, exp_y = vertical_vel_exp[i][:, 0], vertical_vel_exp[i][:, 1]
            exp_x = exp_x + dist if param == 'vel' else exp_x / 0.5 + dist
        else:
            exp_x, exp_y = [], []

        ax.plot(rsm_x, rsm_y, c='r', lw=1.5, ls='-', label='Cal')
        ax.plot(exp_x, exp_y, c='g', lw=0., label='Exp',
                markersize=4, marker='o', markeredgecolor='g',
                markeredgewidth=1.)

    ax.tick_params(labelsize=15, colors='k', direction='in', which='both',
                   width=1., top=True, bottom=True, left=True, right=True)
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[:2], labels[:2], loc="lower right", prop=ppt.font15,
    #           edgecolor='None', frameon=False, labelspacing=0.4,
    #           bbox_to_anchor=(1., -0.7), bbox_transform=ax.transAxes)
    ax.set_aspect("equal")
    ax.grid(visible=True, which='major', axis='x', alpha=1.,
            c='k', ls=':', lw=1.)
    plt.savefig(f"../outputs/cbl_{param}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def cbl_vertical_inflow_profile():
    # from matplotlib import rcParams
    # rcParams['text.usetex'] = True
    # rcParams['font.size'] = 18

    vertical_exp, vertical_rsm = [], []
    for param in ['vel', 'turb']:
        exp_data = f'{data_dir}/inflow_{param}_exp.txt'
        rsm_data = f'{data_dir}/inflow_{param}_rsm.txt'
        vertical_exp.append(np.loadtxt(exp_data, skiprows=4))
        vertical_rsm.append(np.loadtxt(rsm_data, skiprows=4))

    fig, axarray = plt.subplots(1, 2, sharey=True, figsize=(8, 5), dpi=120)
    axs = axarray.flatten()
    axs[0].set_ylabel(r'$z/\delta$', ppt.font18t)
    axs[0].set_ylim([0., 1.])
    axs[0].set_yticks(0.2 * np.arange(6))
    axs[0].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    for i in range(len(axs)):
        ax = axs[i]
        ax.plot(vertical_rsm[i][:, 0], vertical_rsm[i][:, 1],
                c='r', lw=1.5, ls='-', label='Cal')
        ax.plot(vertical_exp[i][:, 0], vertical_exp[i][:, 1],
                c='g', lw=0., label='Exp', markersize=8,
                marker='o', markeredgecolor='g',
                markeredgewidth=1.)
        if i == 0:
            ax.set_xlabel(r'$U/U_0$', ppt.font18t)
            ax.set_xlim([0.5, 1.1])
            ax.set_xticks([0.5, 0.7, 0.9, 1.1])
            ax.set_xticklabels(['0', '0.7', '0.9', '1.1'])
            ax.text(1.0, 0.1, '(a)', va='top', ha='left',
                     fontdict=ppt.font20, )
        else:
            ax.set_xlabel(r'$\sigma_u/U_0$', ppt.font18t)
            ax.set_xlim([0., 0.15])
            ax.set_xticks([0., 0.05, 0.10, 0.15])
            ax.set_xticklabels(['0', '0.05', '0.10', '0.15'])
            ax.text(0.125, 0.1, '(b)', va='top', ha='left',
                    fontdict=ppt.font20, )

        ax.tick_params(labelsize=15, colors='k', direction='in', which='both',
                    width=1., top=True, bottom=True, left=True, right=True)
        tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
        [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    # handles, labels = axs[0].get_legend_handles_labels()
    # axs[0].legend(handles, labels, prop=ppt.font15, edgecolor='None', frameon=False,
    #               ncol=1, labelspacing=0.3, bbox_to_anchor=(0.5, 0.99),
    #               bbox_transform=axs[0].transAxes, columnspacing=0.8, )
    plt.savefig("../outputs/cbl_inflow.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def scale_func(seqs, a, b):
    return ((seqs - np.min(seqs)) / (np.max(seqs) - np.min(seqs))) * (b - a) + a


if __name__ == "__main__":
    cbl_vertical_velocity_profile('vel')
    # cbl_vertical_inflow_profile()