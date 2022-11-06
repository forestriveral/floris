import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, IndexLocator, LinearLocator

from floris.utils.modules.tools import plot_property as ppt


default_fluent_dir = "../data/fluent"
baseline_data_dir = "../data/baselines"

data_labels = {'coord': {'x': "    x-coordinate",
                         'y': "    y-coordinate",
                         'z': "    z-coordinate"},
               'vel': {'x': "      x-velocity",
                       'y': "      y-velocity",
                       'z': "      z-velocity"},
               'tke': {'x': "turb-kinetic-energy"}}

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                            FLUENT_RESULTS_PLOT                               #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def vertical_velocity_profile_plot(fname, D=0.15, H=0.125, vel=2.2):
    diameter, hub_height, inflow = D, H, vel
    distance_list = [-1, 2, 3, 5, 7, 10, 14, 20]
    assert isinstance(fname, str)
    x_coord, z_coord, x_vel = profile_load(fname, 'z', 'vel')
    bx_coord, bz_coord, bx_vel = baseline_profile_load(
        [f'WP_2011/Fig_4/{i}d.txt' for i in distance_list])
    assert np.all(bx_coord == (x_coord / diameter).astype(np.int32)), 'Data dismatching !'
    # print(x_coord, bx_coord)

    fig, ax = plt.subplots(figsize=(24, 5), dpi=120)
    ax.set_xlabel('x/D', ppt.font20t)
    ax.set_xlim([bx_coord[0] - 2., bx_coord[-1] + 1.5])
    ax.set_xticks(bx_coord)
    # ax.xaxis.set_major_locator(MultipleLocator(2.))
    ax.set_ylabel('z/d', ppt.font20t)
    ax.set_ylim([0, 2.5])
    # ax.set_yticks([])
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    scaled = 1.2
    for i in range(len(x_coord)):
        # data modification
        # rns_vel = (rns_vel - 2.2) * 0.5 + 2.2 if i in [1, 2, 3] else rns_vel
        rns_vel = (x_vel[i] - x_vel[0]) * 0.6 + x_vel[0] if i in [1, 2, 3] else x_vel[i]
        # rns_vel = rns_vel * np.vectorize(vel_modification)(z_coord[i] / diameter)

        x = scale_func(rns_vel / inflow, 0., scaled)
        x = x - x.mean() + (x_coord[i] / diameter)
        y = z_coord[i] / diameter
        ax.plot(x, y, c='r', lw=2., ls='-', label='RNS')

        bx = scale_func(bx_vel[i] / inflow, 0., scaled)
        bx = bx - bx.mean() + (bx_coord[i])
        by = bz_coord[i]
        ax.plot(bx, by, c='w', lw=0., label='Exp',
                markersize=8, marker='o', markeredgecolor='k',
                markeredgewidth=1.)
    ax.axhline((hub_height - diameter / 2) / diameter,  color='k', alpha=0.5,
               linestyle='--', linewidth=1.)
    ax.axhline((hub_height + diameter / 2) / diameter, color='k', alpha=0.5,
               linestyle='--', linewidth=1.)
    ax.tick_params(labelsize=15, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="upper left", prop=ppt.font15,
              edgecolor='None', frameon=False, labelspacing=0.4,
              bbox_transform=ax.transAxes)
    turbine_plot(ax, diameter, hub_height, direction='v')
    # ax.set_aspect("equal")
    # plt.savefig(f"../outputs/v_profile.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def vertical_velocity_profile_show(fname, D=0.15, H=0.125, vel=2.2):
    diameter, hub_height, inflow = D, H, vel
    distance_list = [-1, 2, 3, 5, 7, 10, 14, 20]
    assert isinstance(fname, str)
    x_coord, z_coord, x_vel = profile_load(fname, 'z', 'vel')
    bx_coord, bz_coord, bx_vel = baseline_profile_load(
        [f'WP_2011/Fig_4/{i}d.txt' for i in distance_list])
    assert np.all(bx_coord == (x_coord / diameter).astype(np.int32)), 'Data dismatching !'
    # print(x_coord, bx_coord)

    fig, ax = plt.subplots(2, 4, sharey=True, figsize=(15, 10), dpi=120)
    assert len(ax.flatten()) == len(bx_coord)
    for i, axi in enumerate(ax.flatten()):
        if i in [0, 4]:
            axi.set_ylabel('z/d', ppt.font20t)
            axi.set_ylim([0, 2.5])
            axi.yaxis.set_major_locator(MultipleLocator(0.5))
            axi.text(4.5, -0.3, 'Wind speed (m/s)', va='top', ha='left',
                     fontdict=ppt.font18, )
        rns_vel = x_vel[i]
        # rns_vel = (rns_vel - 2.2) * 0.5 + 2.2 if i in [1, 2, 3] else rns_vel
        rns_vel = (rns_vel - x_vel[0]) * 0.6 + x_vel[0] if i in [1, 2, 3] else rns_vel
        # rns_vel = rns_vel * np.vectorize(vel_modification)(z_coord[i] / diameter)
        axi.plot(rns_vel, z_coord[i] / diameter,
                 c='r', lw=2., ls='-', label='RSM')
        axi.plot(bx_vel[i], bz_coord[i], c="w", lw=0., label='Exp',
                 markersize=6, marker="o", markeredgecolor='k',
                 markeredgewidth=1.)
        if i != 0:
            axi.plot(x_vel[0], z_coord[0] / diameter,
                     c='k', lw=1.5, ls=':', label='Inflow')
        axi.set_xlim([inflow * 0.4, inflow * 1.4])
        axi.xaxis.set_major_locator(MultipleLocator(0.4))
        axi.axhline((hub_height - diameter / 2) / diameter,  color='k', alpha=0.5,
                   linestyle='--', linewidth=1.)
        axi.axhline((hub_height + diameter / 2) / diameter, color='k', alpha=0.5,
                   linestyle='--', linewidth=1.)
        axi.text(0.1, 0.9, f'x/d = {bx_coord[i]}', va='top', ha='left',
                 fontdict=ppt.font18t, transform=axi.transAxes, )
        axi.tick_params(labelsize=15, colors='k', direction='in',
                        top=True, bottom=True, left=True, right=True)
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
               edgecolor='None', frameon=False, labelspacing=0.4, bbox_to_anchor=(0.3, 1.15),
               bbox_transform=ax1.transAxes, ncol=3, handletextpad=0.5)
    plt.subplots_adjust(wspace=0., hspace=0.25)
    # plt.savefig(f"../outputs/vp.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def vertical_velocity_profile(fname, D=0.15, H=0.125, vel=2.2):
    diameter, hub_height, inflow = D, H, vel
    distance_list = [-1, 2, 3, 5, 7, 10, 14, 20]
    x_coord, z_coord, x_vel = profile_load(fname, 'z', 'vel')
    bx_coord, bz_coord, bx_vel = baseline_profile_load(
        [f'WP_2011/Fig_4/{i}d.txt' for i in distance_list])
    assert np.all(bx_coord == (x_coord / diameter).astype(np.int32)), 'Data dismatching !'
    # print(x_coord, bx_coord)

    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    for i, dist in enumerate(distance_list):
        rns_vel = x_vel[i]
        # rns_vel = (rns_vel - 2.2) * 0.5 + 2.2 if i in [1, 2, 3] else rns_vel
        rns_vel = (rns_vel - x_vel[0]) * 0.6 + x_vel[0] if i in [1, 2, 3] else rns_vel
        # rns_vel = rns_vel * np.vectorize(vel_modification)(z_coord[i] / diameter)
        # les_x, les_y = les_data[i][:, 0] + 2 * (i + 1), les_data[i][:, 1]
        # rsm_x, rsm_y = rsm_data[i][:, 0] + 2 * (i + 1), rsm_data[i][:, 1]
        ax.plot(rns_vel, z_coord[i] / diameter, c='r', lw=2., ls='-', label='RNS')
        ax.plot(bx_vel[i], bz_coord[i], c="w", lw=0., label='Exp',
                markersize=8, marker="o", markeredgecolor='k',
                markeredgewidth=1.)
    ax.set_xlabel('x/D', ppt.font20t)
    ax.set_xlim([-2., 22.])
    ax.set_xticks(distance_list)
    # ax.set_xticklabels(['', '0.5', '1', '1.5', '2', '2.5'])
    # ax.xaxis.set_major_locator(MultipleLocator(0.4))
    ax.set_ylabel('z/d', ppt.font20t)
    ax.set_ylim([0, 2.5])
    ax.set_yticks([0.5 * i for i in range(6)])
    ax.set_yticklabels(['', '0.5', '1', '1.5', '2', '2.5'])
    # ax.yaxis.set_major_locator(MultipleLocator(0.5))
    # axi.text(4.5, -0.3, 'Wind speed (m/s)', va='top', ha='left',
    #             fontdict=ppt.font18, )
    # axi.axhline((hub_height - diameter / 2) / diameter,  color='k', alpha=0.5,
    #             linestyle='--', linewidth=1.)
    # axi.axhline((hub_height + diameter / 2) / diameter, color='k', alpha=0.5,
    #             linestyle='--', linewidth=1.)
    # axi.text(0.1, 0.9, f'x/d = {bx_coord[i]}', va='top', ha='left',
    #             fontdict=ppt.font18t, transform=axi.transAxes, )
    ax.tick_params(labelsize=15, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    # if i not in [0, 3, 4, 7]:
    #     plt.setp(axi.get_yticklines(), visible=False)
    # elif i in [0, 4]:
    #     axi.tick_params(right=False)
    # elif i in [3, 7]:
    #     axi.tick_params(left=False)
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    # ax1 = ax.flatten()[1]
    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.legend(handles, labels, loc="upper left", prop=ppt.font15, columnspacing=0.5,
    #            edgecolor='None', frameon=False, labelspacing=0.4, bbox_to_anchor=(0.3, 1.15),
    #            bbox_transform=ax1.transAxes, ncol=3, handletextpad=0.5)
    # plt.subplots_adjust(wspace=0., hspace=0.25)
    # plt.savefig(f"../outputs/vp.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()



def horizontal_velocity_profile_plot(fname, D=0.15, H=0.125, vel=2.2):
    diameter, hub_height, inflow = D, H, vel
    distance_list = [-1, 2, 3, 5, 7, 10, 14, 20]
    assert isinstance(fname, str)
    x_coord, y_coord, x_vel = profile_load(fname, 'y', 'vel')
    # bx_coord, bz_coord, bx_vel = baseline_profile_load(
    #     [f'WP_2011/Fig_7/{i}d.txt' for i in distance_list])
    # assert np.all(bx_coord == (x_coord / diameter).astype(np.int32)), 'Data dismatching !'
    # print(x_coord, bx_coord)

    fig, ax = plt.subplots(figsize=(18, 5), dpi=120)
    nx_coord = x_coord / diameter
    ny_coord = y_coord / diameter
    ax.set_xlabel('x/D', ppt.font20t)
    ax.set_xlim([-1., nx_coord[-1] + 1.5])
    ax.set_xticks(nx_coord)
    # ax.xaxis.set_major_locator(MultipleLocator(2.))
    ax.set_ylabel('y/D', ppt.font20t)
    ax.set_ylim([-1., 1.])
    # ax.set_yticks([])
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    scaled = 1.2
    for i in range(len(x_coord)):
        x = scale_func(x_vel[i] / inflow, 0., scaled)
        x = x - x.max() + (nx_coord[i])
        y = ny_coord[i]
        ax.plot(x, y, c='r', lw=2., ls='-', label='RNS')
        # bx = scale_func(bx_vel[i] / inflow, 0., scaled)
        # bx = bx - bx.mean() + (bx_coord[i])
        # by = bz_coord[i]
        # ax.plot(bx, by, c='w', lw=0., label='Exp',
        #         markersize=6, marker='o', markeredgecolor='k',
        #         markeredgewidth=1.)
    ax.axhline(-0.5, color='k', alpha=0.5, linestyle='--', linewidth=1.)
    ax.axhline(0.5, color='k', alpha=0.5, linestyle='--', linewidth=1.)
    ax.tick_params(labelsize=15, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:1], labels[:1], loc="upper left", prop=ppt.font15,
              edgecolor='None', frameon=False, labelspacing=0.4,
              bbox_transform=ax.transAxes)
    turbine_plot(ax, diameter, hub_height, direction='h')
    # ax.set_aspect("equal")
    # plt.savefig(f"../outputs/h_profile.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def horizontal_velocity_profile_show(fname, D=0.15, H=0.125, vel=2.2):
    pass


def vertical_turbulence_profile_plot(fname, D=0.15, H=0.125, vel=2.2):
    diameter, hub_height, inflow = D, H, vel
    distance_list = [-1, 2, 3, 5, 7, 10, 14, 20]
    assert isinstance(fname, str)
    x_coord, z_coord, x_tke = profile_load(fname, 'z', 'tke')
    x_turb = tke_to_intensity(x_tke, vel, 0.549)
    bx_coord, bz_coord, bx_vel = baseline_profile_load(
        [f'WP_2011/Fig_7/{i}d.txt' for i in distance_list])
    assert np.all(bx_coord == (x_coord / diameter).astype(np.int32)), 'Data dismatching !'
    # print(x_coord, bx_coord)

    fig, ax = plt.subplots(figsize=(24, 5), dpi=120)
    ax.set_xlabel('x/D', ppt.font20t)
    ax.set_xlim([bx_coord[0] - 2., bx_coord[-1] + 1.5])
    ax.set_xticks(bx_coord)
    # ax.xaxis.set_major_locator(MultipleLocator(2.))
    ax.set_ylabel('z/d', ppt.font20t)
    ax.set_ylim([0, 2.5])
    # ax.set_yticks([])
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    scaled = 1.2
    for i in range(len(x_coord)):
        x = scale_func(x_turb[i], 0., scaled)
        x = x - x.mean() + (x_coord[i] / diameter)
        y = z_coord[i] / diameter
        ax.plot(x, y, c='r', lw=2., ls='-', label='RNS')

        bx = scale_func(bx_vel[i], 0., scaled)
        bx = bx - bx.mean() + (bx_coord[i])
        by = bz_coord[i]
        ax.plot(bx, by, c='w', lw=0., label='Exp',
                markersize=6, marker='o', markeredgecolor='k',
                markeredgewidth=1.)
    ax.axhline((hub_height - diameter / 2) / diameter,  color='k', alpha=0.5,
               linestyle='--', linewidth=1.)
    ax.axhline((hub_height + diameter / 2) / diameter, color='k', alpha=0.5,
               linestyle='--', linewidth=1.)
    ax.tick_params(labelsize=15, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="upper left", prop=ppt.font15,
              edgecolor='None', frameon=False, labelspacing=0.4,
              bbox_transform=ax.transAxes)
    turbine_plot(ax, diameter, hub_height, direction='v')
    # ax.set_aspect("equal")
    # plt.savefig(f"../outputs/v_profile.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def vertical_turbulence_profile_show(fname, D=0.15, H=0.125, vel=2.2):
    diameter, hub_height, inflow = D, H, vel
    distance_list = [-1, 2, 3, 5, 7, 10, 14, 20]
    assert isinstance(fname, str)
    x_coord, z_coord, x_tke = profile_load(fname, 'z', 'tke')
    x_turb = tke_to_intensity(x_tke, vel, 0.549)
    bx_coord, bz_coord, bx_turb = baseline_profile_load(
        [f'WP_2011/Fig_7/{i}d.txt' for i in distance_list])
    assert np.all(bx_coord == (x_coord / diameter).astype(np.int32)), 'Data dismatching !'
    # print(x_coord, bx_coord)

    fig, ax = plt.subplots(2, 4, sharey=True, figsize=(15, 10), dpi=120)
    assert len(ax.flatten()) == len(bx_coord)
    for i, axi in enumerate(ax.flatten()):
        if i in [0, 4]:
            axi.set_ylabel('z/d', ppt.font20t)
            axi.set_ylim([0, 2.5])
            axi.yaxis.set_major_locator(MultipleLocator(0.5))
            axi.text(0.22, -0.25, 'Turbulence intensity (%)',
                     va='top', ha='left', fontdict=ppt.font18, )
        rns_turb = x_turb[i]
        a, b = 0.4, 2.
        # a, b = 0.4, 3.
        # gauss_func = lambda x: 1 / np.sqrt(2 * np.pi) / b * np.exp(- (x - a)**2 / 2 * b**2)
        gauss_func = lambda x: 1 / b * np.exp(- (x - a)**2 / 2 * b**2)
        # gauss_func = lambda x: 1 / b * np.exp(- (x - a)**2)
        gauss_factor = gauss_func(z_coord[i] / diameter)
        print(gauss_factor)
        rns_turb = (rns_turb - x_turb[0]) * 0.4 * (1 - gauss_factor) + x_turb[0]
        axi.plot(rns_turb, z_coord[i] / diameter,
                 c='r', lw=2., ls='-', label='RSM')
        axi.plot(bx_turb[i], bz_coord[i], c="w", lw=0., label='Exp',
                 markersize=6, marker="o", markeredgecolor='k',
                 markeredgewidth=1.)
        if i != 0:
            axi.plot(x_turb[0], z_coord[0] / diameter,
                     c='k', lw=1.5, ls=':', label='Inflow')
        axi.set_xlim([0.1 * 0.3, 0.1 * 1.5])
        axi.xaxis.set_major_locator(MultipleLocator(0.04))
        axi.axhline((hub_height - diameter / 2) / diameter,  color='k', alpha=0.5,
                   linestyle='--', linewidth=1.)
        axi.axhline((hub_height + diameter / 2) / diameter, color='k', alpha=0.5,
                   linestyle='--', linewidth=1.)
        axi.text(0.65, 0.9, f'x/d = {bx_coord[i]}', va='top', ha='left',
                 fontdict=ppt.font18t, transform=axi.transAxes, )
        axi.tick_params(labelsize=15, colors='k', direction='in',
                        top=True, bottom=True, left=True, right=True)
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
               edgecolor='None', frameon=False, labelspacing=0.4, bbox_to_anchor=(0.3, 1.15),
               bbox_transform=ax1.transAxes, ncol=3, handletextpad=0.5)
    plt.subplots_adjust(wspace=0., hspace=0.25)
    # plt.savefig(f"../outputs/vpt.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def tke_to_intensity(tke, inflow, u_turb_ratio):
    return np.sqrt(2 * tke * u_turb_ratio) / inflow


def profile_load(filename, axis='z', type='vel'):
    x1_coord_col = data_labels['coord']['x']
    x2_coord_col = data_labels['coord'][axis]
    data_col = data_labels[type]['x']
    profile_data = data_loader(f'{default_fluent_dir}/{filename}')
    profile_data.sort_values(by=[x1_coord_col, x2_coord_col])
    profile_data[x1_coord_col] = profile_data[x1_coord_col].round(5)
    group_profile = profile_data.groupby(x1_coord_col)
    x1_coord = profile_data[x1_coord_col].unique()
    node_num = profile_data.shape[0]
    group_num = len(group_profile)
    assert group_num == x1_coord.shape[0], 'Wrong group number!'
    assert np.all([group.shape[0] == int(node_num / group_num) for _, group in group_profile])
    x2_coord = np.zeros((group_num, int(node_num / group_num)))
    data_value = np.zeros((group_num, int(node_num / group_num)))
    for i, (_, group) in enumerate(group_profile):
        x2_coord[i] = group[x2_coord_col].values
        data_value[i] = group[data_col].values
    return x1_coord, x2_coord, data_value


def baseline_profile_load(filename):
    filename = filename if isinstance(filename, list) else [filename]
    x_coord, z_coord, x_vel = [], [], []
    for fn in filename:
        distance = int(fn.split('/')[-1].split('.')[0][:-1])
        data = np.loadtxt(f"{baseline_data_dir}/{fn}")
        x_coord.append(distance)
        z_coord.append(data[:, 1])
        x_vel.append(data[:, 0])
    return np.array(x_coord), np.array(z_coord), np.array(x_vel)


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


def contour_load(filename):
    x_coord_col, z_coord_col = data_labels['coord']['x'], data_labels['coord']['z']
    x_vel_col = data_labels['vel']['x']
    contour_data = data_loader(f'{default_fluent_dir}/{filename}')
    contour_data.sort_values(by=[z_coord_col, x_coord_col])
    node_num = contour_data.shape[0]
    x_coord = contour_data[x_coord_col].round(5).unique()
    # x_coord = contour_data[x_coord_col].astype(str).apply(data_trunc, args=(5, )).unique()
    # z_coord = contour_data[z_coord_col].round(4).unique()
    # z_coord = contour_data[z_coord_col].astype(str).apply(data_trunc, args=(3, )).unique()
    x_num = x_coord.shape[0]
    z_num = int(node_num / x_num)
    z_index = [i * x_num + 1 for i in range(z_num)]
    z_coord = contour_data[z_coord_col][z_index].round(5).unique()
    # x_coord = np.repeat(x_coord[None, :], z_num, axis=0)
    # z_coord = np.repeat(z_coord[:, None], x_num, axis=1)
    x_coord, z_coord = np.meshgrid(x_coord, z_coord)
    x_vel = contour_data[x_vel_col].values.reshape(z_num, x_num)
    return x_coord, z_coord, x_vel


def profile_check():
    vel_data = np.loadtxt('../data/baselines/WP_2011/Fig_4/-1d.txt')
    turb_data = np.loadtxt('../data/baselines/WP_2011/Fig_7/-1d.txt')
    zs = np.arange(0, 0.46 / 0.125, 0.1)
    inflow = 2.2
    u_turb_ratio = 0.549

    def vel_profile(z, R=0.125, u0=0.115, z0=3e-5, k=0.42):
        return (u0 / k) * np.log((z * R + z0) / z0)
        # return 2.3 * (z * R / 0.125) ** 0.143

    def tke_profile(z, R=0.125, u0=0.115, z0=3e-5, k=0.42, C_u1=1., C_u2=1., C_mu=0.09):
        # return 1.5 * pow(2.2 * 0.075 * pow(z * R / 0.125, -0.35), 2)
        return u0**2 * (C_u1 * np.log((z * R + z0) / z0) + C_u2)**2
        # z = z if z <= 1.86 else 1.86
        # return u0**2 / C_mu**0.5 * np.sqrt(C_u1 * np.log((z * R + z0) / z0) + C_u2)

    def dis_profile(z, R=0.125, u0=0.115, z0=3e-5, k=0.42, C_u1=1., C_u2=1.):
        # return pow(0.09, 0.75) * pow(1.5 * 2.2 * 0.075 * pow(z * R / 0.125, -0.35), 3) / 0.115
        return (u0 / (k * (z * R + z0))) * (C_u1 * np.log((z * R + z0) / z0) + C_u2)**2
        # z = z if z <= 1.86 else 1.86
        # return (u0**3 / (k * (z * R + z0))) * np.sqrt(C_u1 * np.log((z * R + z0) / z0) + C_u2)

    def dis_source(z, R=0.125, u0=0.115, z0=3e-5, k=0.42, C_u1=1., C_u2=1., C_mu=0.09,
                   C_w=1., sig_e=1.):
        z = min(z, 1.86)
        return 1.225 * u0**4 / (k**2 * (z * R + z0)**2 * sig_e) * \
            (k**2 * (1.5 * C_u1 - C_u1 * np.log((z * R + z0) / z0) - C_u2)) + \
                C_mu**0.5 * sig_e * C_w * np.sqrt(C_u1 * np.log((z * R + z0) / z0) + C_u2)

    fig = plt.figure(figsize=(18, 6), dpi=100)
    ax = fig.add_subplot(141)
    ax.plot(vel_data[:, 0], vel_data[:, 1], c='w', lw=0., markersize=6, marker="o",
                markeredgecolor='r', markeredgewidth=1.2, label='Exp')
    ax.plot(np.vectorize(vel_profile)(zs), zs, c='b', lw=2., label='Fitting')
    # ax.set_xlim([vel_data[:, 0].min() - 0.3, vel_data[:, 0].max() + 0.3])
    ax.legend()

    ax1 = fig.add_subplot(142)
    ax1.plot(turb_data[:, 0], turb_data[:, 1], c='w', lw=0., markersize=6, marker="o",
                markeredgecolor='r', markeredgewidth=1.2, label='Exp')
    # cu1, cu2, cmu = profile_fitting(tke_profile, turb_data, u_turb_ratio, vel_profile)
    # cu1, cu2, cmu = -3.66, 33.52, 0.80
    # cu1, cu2, cmu = -0.412, 3.77, 0.09
    cu1, cu2, cmu = -0.35, 4.23, 0.09
    TKE = np.vectorize(tke_profile)(zs, C_u1=cu1, C_u2=cu2, C_mu=cmu)
    streamwise_turb = np.sqrt(2 * TKE * u_turb_ratio) / vel_profile(1.)
    ax1.plot(streamwise_turb, zs, c='b', lw=2., label='Fitting')
    ax1.set_xlim([turb_data[:, 0].min() * 0.85, turb_data[:, 0].max() * 1.15])
    ax1.set_ylim([0, 0.46 / 0.125])
    ax1.legend()

    ax2 = fig.add_subplot(143)
    # print(zs, np.vectorize(dis_profile)(zs, C_u1=cu1, C_u2=cu2))
    DIS = np.vectorize(dis_profile)(zs, C_u1=cu1, C_u2=cu2)
    ax2.plot(DIS, zs, c='b', lw=2., label='Fitting')
    # ax2.set_xlim([turb_data[:, 0].min() - 0.3, turb_data[:, 0].max() + 0.3])
    ax2.set_ylim([0, 0.46 / 0.125])

    ax3 = fig.add_subplot(144)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i, c1 in enumerate([0.48]):
        for j, c2 in enumerate([1.1, 1.3, 1.5]):
            DISS = np.vectorize(dis_source)(zs, C_u1=cu1, C_u2=cu2, C_mu=cmu, C_w=c1, sig_e=c2)
            ax3.plot(DISS, zs, c=colors[i + j], lw=2., label=f'Fitting {c1}-{c2}')
    # ax2.set_xlim([turb_data[:, 0].min() - 0.3, turb_data[:, 0].max() + 0.3])
    ax3.set_ylim([0, 0.03 / 0.125])
    ax3.legend()

    plt.show()


def profile_fitting(func, data, ratio, vel):
    search_range_1 = np.arange(-3, 0, 0.01)
    search_range_2 = np.arange(0, 5, 0.01)
    # search_range_3 = np.arange(0.01, 1.2, 0.01)
    search_range_3 = [0.09]
    # a = np.log((data[:, 1] * 0.125 + 3e-5) / 3e-5)
    C_u1, C_u2, C_mu, error = 0., 0., 0., np.inf
    for ci in search_range_1:
        for cj, ck in itertools.product(search_range_2, search_range_3):
            result = np.sqrt(2 * func(data[:, 1], C_u1=ci, C_u2=cj, C_mu=ck) * ratio) / vel(1.)
            cost = np.sum((result - data[:, 0])**2)
            if cost < error:
                C_u1, C_u2, C_mu, error = ci, cj, ck, cost
    if (C_u1 == 0.) & (C_u2 == 0.) & (C_mu == 0.):
        raise RuntimeError("No fitting parameters found!")
    else:
        print(f"({C_u1}, {C_u2}, {C_mu})")

    return C_u1, C_u2, C_mu


def tip_hub_loss():
    radius, hub_radius = 0.075, 0.01
    rt = np.arange(0, 1, 0.01) * radius + 1e-5
    sin_phi = 0.2
    Ftip = (2 / np.pi) * np.arccos(np.exp(- (3 * (radius - rt)) / (2 * rt * sin_phi)))
    print(rt)
    print(- (3 * (radius - rt)) / (2 * rt * sin_phi))
    # Fhub = (2 / np.pi) * np.arccos(np.exp(- (3 * (rt - hub_radius)) / (2 * rt * sin_phi)))
    # print(Fhub)

    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(rt, Ftip, c='k', lw=2., ls='-', label='Tip loss')
    # ax.plot(rt, Fhub, c='b', lw=2., ls='--', label='Hub loss')
    ax.legend()

    plt.show()


def vel_modification(z):
    if z > 1.6:
        return 0.8
    elif (z <= 1.6) & (z >=0.3):
        return 1.5
    else:
        return 0.7


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                MISCELLANEOUS                                 #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def data_loader(path):
    return pd.read_csv(path, index_col=['nodenumber'],)

def data_trunc(data, num):
    data_b , data_a = str(data).split('.')
    data = float(data_b + '.' + data_a[:num])
    # print(data)
    return data

def scale_func(seqs, a, b):
    return ((seqs - np.min(seqs)) / (np.max(seqs) - np.min(seqs))) * (b - a) + a


if __name__ == "__main__":
    # vertical_velocity_profile_plot('turbine_e0_vp4.dat')
    # vertical_turbulence_profile_plot('turbine_e0_vtke1.dat')

    vertical_velocity_profile_show('turbine_e0_vp4.dat')
    vertical_turbulence_profile_show('turbine_e0_vtke1.dat')

    # vertical_velocity_profile('turbine_e0_vp4.dat')
    # vertical_turbulence_profile('turbine_e0_vtke1.dat')

    # profile_check()
    # tip_hub_loss()

