import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, IndexLocator, LinearLocator

from floris.utils.module.tools import plot_property as ppt


data_dir = "../data/others/Lin_yawed"



def horizontal_velocity_profile(output=False):
    distance_list = [4, 6, 8, 10]
    vel_exp_file = [f'vel_30_{i}d_exp' for i in distance_list]
    vel_rsm_file = [f'vel_30_{i}d_rsm' for i in distance_list]
    vel_exp_data, vel_rsm_data = [], []
    for i in range(len(distance_list)):
            vel_exp_data.append(np.loadtxt(f'{data_dir}/{vel_exp_file[i]}.txt', skiprows=4))
            vel_rsm_data.append(np.loadtxt(f'{data_dir}/{vel_rsm_file[i]}.txt', skiprows=4))
    # print(vel_exp_data[0], vel_rsm_data[0])

    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(16, 4), dpi=120)
    for i, axi in enumerate(ax.flatten()):
        if i in [0, ]:
            axi.set_ylabel('y/d', ppt.font20t)
            axi.set_ylim([0, 2.5])
            axi.yaxis.set_major_locator(MultipleLocator(0.5))
            # axi.text(0.5, 0.3, 'Velocity deficit', va='top', ha='left',
            #          fontdict=ppt.font18, )
        axi.plot(vel_rsm_data[i][:, 0], vel_rsm_data[i][:, 1],
                 c='r', lw=2., ls='-', label='RSM')
        axi.plot(vel_exp_data[i][:, 0], vel_exp_data[i][:, 1],
                 c="w", lw=0., label='Exp', markersize=8,
                 marker="o", markeredgecolor='k',
                 markeredgewidth=1.)
        axi.set_xlim([-0.1, 0.8])
        axi.set_xticks([0, 0.2, 0.4, 0.6])
        axi.set_xticklabels(['0', '0.2', '0.4', '0.6'])
        # axi.xaxis.set_major_locator(MultipleLocator(0.2))
        axi.set_ylim([-1, 1.5])
        axi.set_yticks([-1, -0.5, 0., 0.5, 1, 1.5])
        axi.set_yticklabels(['-1', '-0.5', '0', '0.5', '1', '1.5'])
        # axi.yaxis.set_major_locator(MultipleLocator(0.5))
        axi.axhline(0.5, color='k', alpha=0.7, linestyle='--', linewidth=1.)
        axi.axhline(-0.5, color='k', alpha=0.8, linestyle='--', linewidth=1.)
        axi.text(0.7, 0.9, f'x/d = {distance_list[i]}', va='top', ha='left',
                 fontdict=ppt.font18t, transform=axi.transAxes, )
        axi.tick_params(labelsize=15, colors='k', direction='in',
                        top=True, bottom=True, left=True, right=True)
        axi.grid(True, alpha=0.4)
        if i not in [0, 3, 4, 7]:
            plt.setp(axi.get_yticklines(), visible=False)
        elif i in [0, 4]:
            axi.tick_params(right=False)
        elif i in [3, 7]:
            axi.tick_params(left=False)
        tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
        [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    ax1 = ax.flatten()[1]
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="upper left", prop=ppt.font15, columnspacing=0.5,
               edgecolor='None', frameon=False, labelspacing=0.4, bbox_to_anchor=(0.55, 1.15),
               bbox_transform=ax1.transAxes, ncol=3, handletextpad=0.5)
    plt.subplots_adjust(wspace=0., hspace=0.25)
    if output:
        plt.savefig("../outputs/Lin_velocity.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def horizontal_turbulence_profile(output=False):
    distance_list = [4, 6, 8, 10]
    turb_exp_file = [f'turb_30_{i}d_exp' for i in distance_list]
    turb_rsm_file = [f'turb_30_{i}d_rsm' for i in distance_list]
    turb_exp_data, turb_rsm_data = [], []
    for i in range(len(distance_list)):
            turb_exp_data.append(np.loadtxt(f'{data_dir}/{turb_exp_file[i]}.txt', skiprows=4))
            turb_rsm_data.append(np.loadtxt(f'{data_dir}/{turb_rsm_file[i]}.txt', skiprows=4))
    # print(turb_exp_data[0], turb_rsm_data[0])

    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(16, 4), dpi=120)
    for i, axi in enumerate(ax.flatten()):
        if i in [0, ]:
            axi.set_ylabel('y/d', ppt.font20t)
            axi.set_ylim([0, 2.5])
            axi.yaxis.set_major_locator(MultipleLocator(0.5))
            # axi.text(0.5, 0.3, 'turbocity deficit', va='top', ha='left',
            #          fontdict=ppt.font18, )
        axi.plot(turb_rsm_data[i][:, 0], turb_rsm_data[i][:, 1],
                 c='r', lw=2., ls='-', label='RSM')
        axi.plot(turb_exp_data[i][:, 0], turb_exp_data[i][:, 1],
                 c="w", lw=0., label='Exp', markersize=8,
                 marker="o", markeredgecolor='k',
                 markeredgewidth=1.)
        axi.set_xlim([0.05, 0.22])
        axi.set_xticks([0.1, 0.15, 0.2])
        axi.set_xticklabels(['0.1', '0.15', '0.2'])
        # axi.xaxis.set_major_locator(MultipleLocator(0.2))
        axi.set_ylim([-1, 1.4])
        axi.set_yticks([-1, -0.5, 0., 0.5, 1])
        axi.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])
        # axi.yaxis.set_major_locator(MultipleLocator(0.5))
        axi.axhline(0.5, color='k', alpha=0.7, linestyle='--', linewidth=1.)
        axi.axhline(-0.5, color='k', alpha=0.8, linestyle='--', linewidth=1.)
        axi.text(0.7, 0.9, f'x/d = {distance_list[i]}', va='top', ha='left',
                 fontdict=ppt.font18t, transform=axi.transAxes, )
        axi.tick_params(labelsize=15, colors='k', direction='in',
                        top=True, bottom=True, left=True, right=True)
        axi.grid(True, alpha=0.4)
        if i not in [0, 3, 4, 7]:
            plt.setp(axi.get_yticklines(), visible=False)
        elif i in [0, 4]:
            axi.tick_params(right=False)
        elif i in [3, 7]:
            axi.tick_params(left=False)
        tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
        [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    ax1 = ax.flatten()[1]
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="upper left", prop=ppt.font15, columnspacing=0.5,
               edgecolor='None', frameon=False, labelspacing=0.4, bbox_to_anchor=(0.55, 1.15),
               bbox_transform=ax1.transAxes, ncol=3, handletextpad=0.5)
    plt.subplots_adjust(wspace=0., hspace=0.25)
    if output:
        plt.savefig("../outputs/Lin_turbulence.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def wake_center_plot(ind='a', output=False):
    data_path = f"../data/baselines/LP_2019/Fig_8/{ind}"
    yaw_angle = [10, 20, 30]
    vel_exp_file = [f'yaw_{i}_exp' for i in yaw_angle]
    vel_rsm_file = [f'yaw_{i}_rsm' for i in yaw_angle]
    vel_exp_data, vel_rsm_data = [], []
    for i in range(len(yaw_angle)):
            vel_exp_data.append(np.loadtxt(f'{data_path}/{vel_exp_file[i]}.txt', skiprows=4))
            vel_rsm_data.append(np.loadtxt(f'{data_path}/{vel_rsm_file[i]}.txt', skiprows=4))
    # print(vel_exp_data[0], vel_rsm_data[0])

    fig, ax = plt.subplots(3, 1, sharey=True, figsize=(8, 8), dpi=120)
    for i, axi in enumerate(ax.flatten()):
        axi.plot(vel_rsm_data[i][:, 0], vel_rsm_data[i][:, 1],
                 c='r', lw=2., ls='-', label='RSM')
        axi.plot(vel_exp_data[i][:, 0], vel_exp_data[i][:, 1],
                 c="w", lw=0., label='Exp', markersize=9,
                 marker="o", markeredgecolor='k',
                 markeredgewidth=1.)
        axi.set_xlim([3., 10.])
        axi.set_xticks(range(3, 11))
        axi.set_xticklabels([str(i) for i in range(3, 11)])
        # axi.xaxis.set_major_locator(MultipleLocator(0.2))
        axi.set_ylim([0., 1.])
        axi.set_yticks([0., 0.5, 1])
        axi.set_yticklabels(['0', '0.5', '1'])
        # axi.yaxis.set_major_locator(MultipleLocator(0.5))
        # axi.axhline(0.5, color='k', alpha=0.7, linestyle='--', linewidth=1.)
        # axi.axhline(-0.5, color='k', alpha=0.8, linestyle='--', linewidth=1.)
        axi.text(0.7, 0.85, f'yaw = {yaw_angle[i]} degree', va='top', ha='left',
                 fontdict=ppt.font18t, transform=axi.transAxes, )
        axi.tick_params(labelsize=15, colors='k', direction='in',
                        top=True, bottom=True, left=True, right=True)
        # axi.grid(True, alpha=0.4)
        # if i not in [0, 3, 4, 7]:
        #     plt.setp(axi.get_yticklines(), visible=False)
        # elif i in [0, 4]:
        #     axi.tick_params(right=False)
        # elif i in [3, 7]:
        #     axi.tick_params(left=False)
        if i in [3, ]:
            axi.set_xlabel('x/d', ppt.font20t)
        tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
        [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    ax1 = ax.flatten()[1]
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="upper left", prop=ppt.font15, columnspacing=0.5,
               edgecolor='None', frameon=False, labelspacing=0.4, bbox_to_anchor=(0.30, 2.55),
               bbox_transform=ax1.transAxes, ncol=3, handletextpad=0.5)
    plt.subplots_adjust(wspace=0.5, hspace=0.25)
    if output:
        plt.savefig(f"../outputs/vel_center_{ind}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # horizontal_velocity_profile(output=True)
    # horizontal_turbulence_profile(output=True)
    wake_center_plot(ind='a', output=True)