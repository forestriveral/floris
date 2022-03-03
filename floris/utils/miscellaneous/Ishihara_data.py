import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, IndexLocator, LinearLocator

from floris.utils.visualization import property as ppt


data_dir = "./Ishihara_nonyawed"


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                            ISHIHARA_DATA_PLOT                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def vv_profile_plot(fcase='137_037_vv', D=0.15, H=0.125, vel=2.2):
    diameter, hub_height, inflow = D, H, vel
    distance_list = [2, 4, 6, 8]
    vel_2d_les = np.loadtxt(f'{data_dir}/{fcase}_2d_les.txt')
    vel_4d_les = np.loadtxt(f'{data_dir}/{fcase}_4d_les.txt')
    vel_6d_les = np.loadtxt(f'{data_dir}/{fcase}_6d_les.txt')
    vel_8d_les = np.loadtxt(f'{data_dir}/{fcase}_8d_les.txt')

    vel_2d_exp = np.loadtxt(f'{data_dir}/{fcase}_2d_exp.txt')
    vel_8d_exp = np.loadtxt(f'{data_dir}/{fcase}_8d_exp.txt')

    vel_2d_rsm = np.loadtxt(f'{data_dir}/{fcase}_2d_rsm.txt')
    vel_4d_rsm = np.loadtxt(f'{data_dir}/{fcase}_4d_rsm.txt')
    vel_6d_rsm = np.loadtxt(f'{data_dir}/{fcase}_6d_rsm.txt')
    vel_8d_rsm = np.loadtxt(f'{data_dir}/{fcase}_8d_rsm.txt')

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    ax.set_xlabel('x/D', ppt.font20t)
    ax.set_xlim([0, 10])
    # ax.set_xticks([-2, 10])
    ax.xaxis.set_major_locator(MultipleLocator(2.))
    ax.set_ylabel('z/H', ppt.font20t)
    ax.set_ylim([0, 2.5])
    ax.set_yticks([0.5 * i for i in range(6)])
    ax.set_yticklabels(['', '0.5', '1', '1.5', '2', '2.5'])
    # ax.yaxis.set_major_locator(MultipleLocator(0.5))

    scaled = 1.
    les_data = [vel_2d_les, vel_4d_les, vel_6d_les, vel_8d_les]
    rsm_data = [vel_2d_rsm, vel_4d_rsm, vel_6d_rsm, vel_8d_rsm]
    exp_data = [vel_2d_exp, None, None, vel_8d_exp]
    for i in range(len(les_data)):
        les_x, les_y = les_data[i][:, 0] + 2 * (i + 1), les_data[i][:, 1]
        rsm_x, rsm_y = rsm_data[i][:, 0] + 2 * (i + 1), rsm_data[i][:, 1]

        ax.plot(rsm_x, rsm_y, c='r', lw=2., ls='-', label='RSM')
        ax.plot(les_x, les_y, c='k', lw=2., ls='-', label='LES')
        # ax.plot(les_x, les_y, c='k', lw=0., label='LES',
        #         markersize=10, marker='x', markeredgecolor='k',
        #         markeredgewidth=1.)

        if exp_data[i] is not None:
            exp_x, exp_y = exp_data[i][:, 0] + 2 * (i + 1), exp_data[i][:, 1]
            ax.plot(exp_x, exp_y, c='w', lw=0., label='Exp',
                    markersize=10, marker='o', markeredgecolor='k',
                    markeredgewidth=1.)

        # les_x, les_y = les_data[:, 0] + 2 * (i + 1), les_data[:, 1]
        # ax.plot(les_x, les_y, c='k', lw=2., ls='-', label='LES')

    # ax.axhline((hub_height - diameter / 2) / diameter,  color='k', alpha=0.5,
    #            linestyle='--', linewidth=1.)
    # ax.axhline((hub_height + diameter / 2) / diameter, color='k', alpha=0.5,
    #            linestyle='--', linewidth=1.)
    ax.tick_params(labelsize=18, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], loc="upper left", prop=ppt.font15,
              edgecolor='None', frameon=False, labelspacing=0.4,
              bbox_transform=ax.transAxes)
    # turbine_plot(ax, diameter, hub_height, direction='v')
    # ax.set_aspect("equal")
    plt.savefig(f"./Ishihara_nonyawed/{fcase}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def tv_profile_plot(fcase='137_037_tv', D=0.15, H=0.125, vel=2.2):
    diameter, hub_height, inflow = D, H, vel
    distance_list = [2, 4, 6, 8]
    turb_2d_les = np.loadtxt(f'{data_dir}/{fcase}_2d_les.txt')
    turb_4d_les = np.loadtxt(f'{data_dir}/{fcase}_4d_les.txt')
    turb_6d_les = np.loadtxt(f'{data_dir}/{fcase}_6d_les.txt')
    turb_8d_les = np.loadtxt(f'{data_dir}/{fcase}_8d_les.txt')

    turb_2d_exp = np.loadtxt(f'{data_dir}/{fcase}_2d_exp.txt')
    turb_8d_exp = np.loadtxt(f'{data_dir}/{fcase}_8d_exp.txt')

    turb_2d_rsm = np.loadtxt(f'{data_dir}/{fcase}_2d_rsm.txt')
    turb_4d_rsm = np.loadtxt(f'{data_dir}/{fcase}_4d_rsm.txt')
    turb_6d_rsm = np.loadtxt(f'{data_dir}/{fcase}_6d_rsm.txt')
    turb_8d_rsm = np.loadtxt(f'{data_dir}/{fcase}_8d_rsm.txt')

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    ax.set_xlabel('x/D', ppt.font20t)
    ax.set_xlim([0, 10])
    # ax.set_xticks([-2, 10])
    ax.xaxis.set_major_locator(MultipleLocator(2.))
    ax.set_ylabel('z/H', ppt.font20t)
    ax.set_ylim([0, 2.5])
    ax.set_yticks([0.5 * i for i in range(6)])
    ax.set_yticklabels(['', '0.5', '1', '1.5', '2', '2.5'])
    # ax.yaxis.set_major_locator(MultipleLocator(0.5))

    scaled = 1.
    les_data = [turb_2d_les, turb_4d_les, turb_6d_les, turb_8d_les]
    rsm_data = [turb_2d_rsm, turb_4d_rsm, turb_6d_rsm, turb_8d_rsm]
    exp_data = [turb_2d_exp, None, None, turb_8d_exp]
    for i in range(len(les_data)):
        les_x, les_y = les_data[i][:, 0] + 2 * (i + 1), les_data[i][:, 1]
        rsm_x, rsm_y = rsm_data[i][:, 0] + 2 * (i + 1), rsm_data[i][:, 1]

        ax.plot(rsm_x, rsm_y, c='r', lw=2., ls='-', label='RSM')
        ax.plot(les_x, les_y, c='k', lw=2., ls='-', label='LES')
        # ax.plot(les_x, les_y, c='k', lw=0., label='LES',
        #         markersize=10, marker='x', markeredgecolor='k',
        #         markeredgewidth=1.)

        if exp_data[i] is not None:
            exp_x, exp_y = exp_data[i][:, 0] + 2 * (i + 1), exp_data[i][:, 1]
            ax.plot(exp_x, exp_y, c='w', lw=0., label='Exp',
                    markersize=10, marker='o', markeredgecolor='k',
                    markeredgewidth=1.)

        # les_x, les_y = les_data[:, 0] + 2 * (i + 1), les_data[:, 1]
        # ax.plot(les_x, les_y, c='k', lw=2., ls='-', label='LES')

    # ax.axhline((hub_height - diameter / 2) / diameter,  color='k', alpha=0.5,
    #            linestyle='--', linewidth=1.)
    # ax.axhline((hub_height + diameter / 2) / diameter, color='k', alpha=0.5,
    #            linestyle='--', linewidth=1.)
    ax.tick_params(labelsize=18, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], loc="upper left", prop=ppt.font15,
              edgecolor='None', frameon=False, labelspacing=0.4,
              bbox_transform=ax.transAxes)
    # turbine_plot(ax, diameter, hub_height, direction='v')
    # ax.set_aspect("equal")
    plt.savefig(f"./Ishihara_nonyawed/{fcase}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def turbine_plot(ax, D, H, direction='v', origin=True):
    x_rotor = [0., 0.]
    y_rotor = [(H - D / 2) / D, (H + D / 2) / D] if direction == 'v' else [-0.5, 0.5]
    ax.plot(x_rotor, y_rotor, c='k', lw=2., ls='-')

    xy_nacelle = (0., H / D - 0.05) if direction == 'v' else [0., -0.01 / D / 2]
    width, height = 0.05 / D, 0.01 / D
    rect = patches.Rectangle(xy_nacelle, width, height, color='k', )
    ax.add_patch(rect)

    if direction == 'v':
        x_tower, y_tower = [0.1, 0.1], [0., H / D]
        ax.plot(x_tower, y_tower, c='k', lw=2.5, ls='-')




if __name__ == "__main__":
    # vv_profile_plot('137_037_vv')
    tv_profile_plot('137_037_tv')