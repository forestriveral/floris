import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from floris.utils.module.tools import plot_property as ppt


data_dir = "../data/others/Blind_test"


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                               NTNU_BLIND_TEST                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def aligned_velocity_profile():
    distance_list = [1, 2.5, 4]
    vel_exp, vel_rsm = [], []
    for dist in distance_list:
        exp_data = f'{data_dir}/aligned/a_{dist}D_vel_exp.txt'
        rsm_data = f'{data_dir}/aligned/a_{dist}D_vel_rsm.txt'
        if os.path.exists(exp_data):
            vel_exp.append(np.loadtxt(exp_data, skiprows=4))
        else:
            vel_exp.append([])
        if os.path.exists(rsm_data):
            vel_rsm.append(np.loadtxt(rsm_data, skiprows=4))
        else:
            vel_rsm.append([])

    fig, ax = plt.subplots(1, len(distance_list), sharex=True, figsize=(15, 5), dpi=120)
    for i, axi in enumerate(ax.flatten()):
        rsm_x, rsm_y = vel_rsm[i][:, 0], vel_rsm[i][:, 1]
        exp_x, exp_y = vel_exp[i][:, 0], vel_exp[i][:, 1]
        axi.plot(rsm_x, rsm_y, c='b', lw=2.5, ls='-', label='RSM')
        axi.plot(exp_x, exp_y, c='w', lw=0., label='Exp',
                markersize=8, marker='o', markeredgecolor='k',
                markeredgewidth=1.)

        axi.set_xlabel('x/D', ppt.font18t)
        axi.set_xlim([-2.5, 2.5])
        axi.set_xticks([-2., -1., 0., 1., 2.])
        # ax.set_xticks(np.arange(0, 12, 0.5), minor=True)
        axi.set_xticklabels(['-2', '-1', '0', '1', '2'])
        # ax.set_xticklabels([str(int(i)) if int(i) == i else '' for i in 0.5 * np.arange(23)])
        if i == 0:
            axi.set_ylabel('Normalized velocity', ppt.font18t)
            handles, labels = axi.get_legend_handles_labels()
            axi.legend(handles[:], labels[:], loc="lower left", prop=ppt.font15,
                      edgecolor='None', frameon=False, labelspacing=0.4,
                      bbox_transform=axi.transAxes)
        axi.set_ylim([0., 1.4])
        axi.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4])
        axi.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1', '1.2', '1.4'])
        axi.text(0.4, 1.1, f'x/D = {distance_list[i]}', va='top', ha='left',
                 fontdict=ppt.font18t, transform=axi.transAxes, )
        axi.tick_params(labelsize=15, colors='k', direction='in', which='both',
                    width=1., top=True, bottom=True, left=True, right=True)
        tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
        [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    plt.savefig("../outputs/blind_aligned.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def staggered_velocity_profile():
    distance_list = [1, 3]
    vel_exp, vel_rsm = [], []
    for dist in distance_list:
        exp_data = f'{data_dir}/staggered/a_{dist}D_vel_exp.txt'
        rsm_data = f'{data_dir}/staggered/a_{dist}D_vel_rsm.txt'
        if os.path.exists(exp_data):
            vel_exp.append(np.loadtxt(exp_data, skiprows=4))
        else:
            vel_exp.append([])
        if os.path.exists(rsm_data):
            vel_rsm.append(np.loadtxt(rsm_data, skiprows=4))
        else:
            vel_rsm.append([])

    fig, ax = plt.subplots(1, len(distance_list), sharex=True, figsize=(12, 6), dpi=120)
    for i, axi in enumerate(ax.flatten()):
        rsm_x, rsm_y = vel_rsm[i][:, 0], vel_rsm[i][:, 1]
        exp_x, exp_y = vel_exp[i][:, 0], vel_exp[i][:, 1]
        axi.plot(rsm_x, rsm_y, c='b', lw=2.5, ls='-', label='RSM')
        axi.plot(exp_x, exp_y, c='w', lw=0., label='Exp',
                markersize=8, marker='o', markeredgecolor='k',
                markeredgewidth=1.)

        axi.set_xlabel('x/D', ppt.font18t)
        axi.set_xlim([-2.5, 2.5])
        axi.set_xticks([-2., -1., 0., 1., 2.])
        # ax.set_xticks(np.arange(0, 12, 0.5), minor=True)
        axi.set_xticklabels(['-2', '-1', '0', '1', '2'])
        # ax.set_xticklabels([str(int(i)) if int(i) == i else '' for i in 0.5 * np.arange(23)])
        if i == 0:
            axi.set_ylabel('Normalized velocity', ppt.font18t)
            handles, labels = axi.get_legend_handles_labels()
            axi.legend(handles[:], labels[:], loc="lower left", prop=ppt.font15,
                      edgecolor='None', frameon=False, labelspacing=0.4,
                      bbox_transform=axi.transAxes)
        axi.set_ylim([0., 1.4])
        axi.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4])
        axi.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1', '1.2', '1.4'])
        axi.text(0.4, 0.93, f'x/D = {distance_list[i]}', va='top', ha='left',
                 fontdict=ppt.font18t, transform=axi.transAxes, )
        axi.tick_params(labelsize=15, colors='k', direction='in', which='both',
                    width=1., top=True, bottom=True, left=True, right=True)
        tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
        [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    plt.savefig("../outputs/blind_staggered.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    aligned_velocity_profile()
    # staggered_velocity_profile()