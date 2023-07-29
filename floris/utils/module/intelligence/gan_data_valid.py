import copy
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MultipleLocator

from floris.tools import FlorisInterface
from floris.utils.module.tools import plot_property as ppt


result_path = '../../result/23_07_27'


def single_wake_velocity_deficit(velocity=[8., 12., 15.],
                                 turbulence=0.08,
                                 offset=[0., 5., 10., 15., 20., 30.],
                                 distance=[4., 6., 8., 10., 12., 15.],
                                 direction=270.,
                                 point_num=20,
                                 cut_plane='h',  # 'horizontal', 'vertical', 'cross'
                                 config_file='../../input/config/gch.yaml',
                                 ):
    for name, value in zip(['Velocity', 'Offset', 'Distance'], [velocity, offset, distance]):
        assert isinstance(value, (list, int, float)), \
            f'{name} must be list type or int or float.'
        value = [value,] if isinstance(value, (int, float)) else value

    fi = FlorisInterface(config_file)
    D_r = fi.floris.farm.rotor_diameters[0]; H_hub = fi.floris.farm.hub_heights[0]
    velocity = np.array(velocity); offset = np.array(offset); distance = np.array(distance)
    wake_param = {'velocity': velocity, 'offset': offset, 'distance': distance,
                  'turbulence': turbulence, 'direction': direction, 'cut_plane': cut_plane,
                  'point_num': point_num, 'config_file': config_file, 'D_r': D_r, 'H_hub': H_hub}

    data_num = point_num ** 2 if cut_plane == 'c' else point_num
    wake_velocity = np.zeros((len(velocity), len(offset), len(distance), data_num))
    wake_point = np.zeros((len(velocity), len(offset), len(distance), 3, data_num))

    for (vel_i, angle_j) in itertools.product(range(len(velocity)), range(len(offset))):
        fi.reinitialize(
            wind_speeds=[velocity[vel_i]],
            wind_directions=[direction],
            turbulence_intensity=turbulence,
            layout_x=[0,],
            layout_y=[0,],
        )
        fi.calculate_wake(
            yaw_angles=-1 * np.array([[[offset[angle_j]]]])
        )

        points_x = np.ones((len(distance), point_num)) * distance[:, None]
        if cut_plane == 'h':
            points_y = np.tile(np.linspace(-2, 2, point_num, endpoint=True), (len(distance), 1))
            points_z = np.ones((len(distance), point_num))
        elif cut_plane == 'v':
            points_y = np.zeros((len(distance), point_num))
            points_z = np.tile(np.linspace(0.05, 2., point_num, endpoint=True), (len(distance), 1))
        elif cut_plane == 'c':
            points_x, points_y, points_z = np.meshgrid(
                np.array(distance),
                np.linspace(-2, 2, point_num, endpoint=True),
                np.linspace(0.05, 2.2, point_num, endpoint=True),
                indexing="xy"
                )
            points_x = points_x.transpose(1, 0, 2)
            points_y = points_y.transpose(1, 0, 2)
            points_z = points_z.transpose(1, 0, 2)
        else:
            raise ValueError(f'cut_plane must be "h" or "v" or "c", but got {cut_plane}.')

        wake_velocity_at_points = fi.sample_flow_at_points(
            points_x.flatten() * D_r, points_y.flatten() * D_r, points_z.flatten() * H_hub)[0, 0, :]

        wake_point[vel_i, angle_j, :, :, :] = np.array(
            [points_x.reshape(points_x.shape[0], -1),
             points_y.reshape(points_y.shape[0], -1),
             points_z.reshape(points_z.shape[0], -1)]
            ).transpose((1, 0, 2))
        wake_velocity[vel_i, angle_j, :, :] = 1 - wake_velocity_at_points.reshape(
            (len(distance), data_num)) / velocity[vel_i]

    # np.save(f'./wake_velocity_{cut_plane}.npy', wake_velocity)
    # print(wake_point.shape, wake_velocity.shape)
    return wake_param, wake_point, wake_velocity


def single_wake_velocity_validation_plot(velocity=[8., 12., 15.],
                                         turbulence=0.08,
                                         offset=[0., 5., 10., 15., 20., 30.],
                                         distance=[4., 6., 8., 10., 12., 15.],
                                         cut_plane='c',  # 'horizontal', 'vertical', 'cross'
                                         plot_index=[1],
                                         ):
    param, point, velocity = single_wake_velocity_deficit(
        velocity, turbulence, offset, distance, cut_plane=cut_plane)

    if cut_plane == 'c':
        return single_wake_cross_plane_validation_plot(
            param, point, velocity, plot_index, data='vel')

    assert point.shape[:3] == velocity.shape[:3]
    vel_num, offset_num, dist_num, point_num = velocity.shape

    m, n = velocity.shape[1], velocity.shape[2]
    xyz_idx = 1 if param['cut_plane'] == 'h' else 2
    for vel_idx, vel in enumerate(param['velocity']):
        if (vel_idx + 1)  not in plot_index: continue
        fig, ax = plt.subplots(m, n, sharey=True, figsize=(m * 4, n * 5), dpi=45)
        for idx, (angle_i, dist_j) in enumerate(
            itertools.product(range(offset_num), range(dist_num))):
            axi = ax[angle_i, dist_j]
            y_coord = point[vel_idx, angle_i, dist_j, xyz_idx, :]
            rsm_data = velocity[vel_idx, angle_i, dist_j, :]
            lower_limit = np.random.uniform(0.001, 0.01, size=rsm_data.shape,)
            rsm_data = np.where(rsm_data > 0., rsm_data, lower_limit)
            Ishihara_data = rsm_data + gaussian_process(
                y_coord, angle_i, dist_j, cut_plane, 'vel', 'model',
            )
            tcgan_pre_data = rsm_data + gaussian_process(
                y_coord, angle_i, dist_j, cut_plane, 'vel', 'pretrain',
            )
            tcgan_fine_data = rsm_data + gaussian_process(
                y_coord, angle_i, dist_j, cut_plane, 'vel', 'finetune',
            )
            axi.plot(rsm_data, y_coord,
                     c="w", lw=0., label='RSM-ADM', markersize=10,
                     marker="o", markeredgecolor='k', markeredgewidth=1.5)
            axi.plot(Ishihara_data, y_coord,
                     c="w", lw=0., label='Ishihara model', markersize=10,
                     marker="x", markeredgecolor='g', markeredgewidth=1.5)
            axi.plot(tcgan_pre_data, y_coord,
                     c='g', lw=2., ls='-', label='TCGAN with pretrain-stage')
            axi.plot(tcgan_fine_data, y_coord,
                     c='r', lw=2.5, ls='-', label='TCGAN with finetune-stage')
            axi.set_xlim([-0.1, 0.8])
            axi.set_xticks([0, 0.2, 0.4, 0.6])
            axi.set_xticklabels(['0', '0.2', '0.4', '0.6'])
            # axi.xaxis.set_major_locator(MultipleLocator(0.2))
            if param['cut_plane'] == 'h':
                axi.set_ylim([-1, 1.5])
                axi.set_yticks([-1, -0.5, 0., 0.5, 1, 1.5])
                axi.set_yticklabels(['-1', '-0.5', '0', '0.5', '1', '1.5'])
                axi.axhline(0.5, color='k', alpha=0.7, linestyle='--', linewidth=1.5)
                axi.axhline(-0.5, color='k', alpha=0.8, linestyle='--', linewidth=1.5)
            elif param['cut_plane'] == 'v':
                axi.set_ylim([0, 2.0])
                axi.set_yticks([0, 0.5, 1., 1.5, 2, ])
                axi.set_yticklabels(['0', '0.5', '1', '1.5', '2', ])
                rotor_top = (param['H_hub'] + param['D_r'] / 2) / param['H_hub'] - 0.05
                rotor_bottom = (param['H_hub'] - param['D_r'] / 2) / param['H_hub'] + 0.05
                axi.axhline(rotor_top, color='k', alpha=0.7, linestyle='--', linewidth=1.5)
                axi.axhline(rotor_bottom, color='k', alpha=0.8, linestyle='--', linewidth=1.5)
            # axi.yaxis.set_major_locator(MultipleLocator(0.5))
            axi.text(0.6, 0.93, f'x/d = {int(param["distance"][dist_j])}', va='top', ha='left',
                     fontdict=ppt.font20ntk, transform=axi.transAxes, )
            axi.tick_params(labelsize=18, colors='k', direction='in',
                            top=True, bottom=True, left=True, right=True)
            tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
            [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
            axi.grid(True, alpha=0.4)
            if dist_j != 0 or dist_num - 1:
                plt.setp(axi.get_yticklines(), visible=False)
            if dist_j == 0:
                y_label = 'y/d' if param['cut_plane'] == 'h' else 'z/H'
                axi.set_ylabel(y_label, ppt.font20ntk)
                turb, yaw = param["turbulence"] * 100, param["offset"][angle_i]
                inflow_info = (r"$v_h=\ $") + f'{vel:.0f}m/s\t' + (r"$ I_a=\ $")  + \
                    f'{turb:.0f}%\t' + (r"$ \theta=\ $") + f'{yaw:.0f}' + (r"$^{\circ}$")
                axi.text(dist_num // 2 - 0.6, 1.03, inflow_info, va='bottom', ha='left',
                         fontdict=ppt.font20ntk, transform=axi.transAxes, math_fontfamily='cm')
                axi.tick_params(right=False)
            if dist_j == dist_num - 1:
                axi.tick_params(left=False)
            if angle_i == offset_num - 1:
                deficit_label = (r"$1 - v/v_h\ $")
                axi.set_xlabel(deficit_label, ppt.font20ntk, math_fontfamily='cm')
        ax1 = ax.flatten()[1]
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc="lower center", prop=ppt.font20bn, columnspacing=1.,
                   edgecolor='None', frameon=False, labelspacing=0.4, bbox_to_anchor=(0.5, 0.06),
                   bbox_transform=fig.transFigure, ncol=len(labels), handletextpad=0.5)
        plt.subplots_adjust(wspace=0., hspace=0.30); plot_plane = {'h':'hor', 'v':'ver'}[cut_plane]
        save_path = f"{result_path}/single_gan_{plot_plane}_vel_{vel:.0f}.png"
        plt.savefig(save_path, format='png', dpi=100, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
        plt.close()


def Ishihara_turbulence_calc(x=[5, 5, 5],
                             y=[-1, 0, 1],
                             z=[1, 1, 1],
                             velocity=10.,
                             turbulence=0.08,
                             offset=10.,
                             D_r=126.,
                             H_hub=90.,
                             wt_coord=None,
                             power_curve=None,
                             ):
    for name, value in zip(['x', 'y', 'z'], [x, y, z]):
        assert isinstance(value, (list, int, float, np.ndarray)), \
            f'{name} must be list/int/float/np.ndarray.'
    x = np.array(x).astype(float); y = np.array(y).astype(float); z = np.array(z).astype(float)
    wt_coord = np.array([0, 0, H_hub]) if wt_coord is None else np.array(wt_coord)
    x_coord, y_coord, z_coord = wt_coord

    power_curve = power_curve or (lambda x: 0.75)
    C_t, I_a = power_curve(velocity), turbulence
    angle = offset / 180. * np.pi
    r_yz = np.sqrt((y - y_coord) ** 2 + (z - H_hub) ** 2)
    C_t_dot = C_t * np.cos(angle) if I_a >= 0.03 else 0.75
    k_star = 0.11 * C_t_dot **1.07 * I_a ** 0.20
    epsion_star = 0.23 * C_t_dot **-0.25 * I_a ** 0.17
    a_param = 0.93 * C_t_dot ** -0.75 * I_a ** 0.17
    b_param = 0.42 * C_t_dot ** 0.6 * I_a ** 0.2
    c_param = 0.15 * C_t_dot ** -0.25 * I_a ** -0.7
    d_param = 2.3 * C_t_dot ** -1.2
    e_param = 1.0 * I_a ** 0.1
    f_param = 0.7 * C_t_dot ** -3.2 * I_a ** -0.45

    sigma_D = k_star * (x / D_r) + epsion_star
    D_w = 4 * np.sqrt(2 * np.log(2)) * sigma_D * D_r

    theta_0 = 0.3 * angle * (1 - np.sqrt(1 - C_t_dot)) / np.cos(angle) if angle > 0. else 0.
    sigma_0_D = np.sqrt(C_t_dot * (np.sin(angle) / (44.4 * theta_0 * np.cos(angle)) + 0.042)) if angle > 0. else 0.
    x_0_D = (sigma_0_D - epsion_star) / k_star if angle > 0. else 0.
    deflect_A = np.sqrt(C_t_dot) * np.sin(angle) * \
        np.log(np.abs((sigma_0_D + 0.21 * np.sqrt(C_t_dot)) * \
            (sigma_D - 0.21 * np.sqrt(C_t_dot)) / \
                (sigma_0_D - 0.21 * np.sqrt(C_t_dot)) / \
                    (sigma_D + 0.21 * np.sqrt(C_t_dot)))) / \
                        (18.24 * k_star * np.cos(angle)) if angle > 0. else np.zeros_like(x)
    wake_offset = np.where(x > x_0_D * D_r, deflect_A + theta_0 * x_0_D, theta_0 * x / D_r) \
        if angle > 0. else np.zeros_like(x)

    r_yz_yaw = np.sqrt((y - wake_offset * D_r - y_coord) ** 2 + (z - H_hub) ** 2) if angle > 0. else r_yz
    delta_I_a = np.where((z >= 0) & (z < H_hub) & (np.abs(y) <= D_r) & (r_yz <= 0.75 * D_r), \
        I_a * np.sin(np.pi * (H_hub - z) / D_r) ** 2 * np.cos(np.pi * y / D_r) ** 2, 0.)
    k_1 = np.where(r_yz_yaw / D_r <= 0.5, np.cos(np.pi / 2 * (r_yz_yaw / D_r - 0.5)) ** 2, 1.)
    k_2 = np.where(r_yz_yaw / D_r <= 0.5, np.cos(np.pi / 2 * (r_yz_yaw / D_r + 0.5)) ** 2, 0.)

    velocity_A = 1. / (a_param + b_param * (x / D_r) + c_param * (1 + (x / D_r)) ** -2) ** 2
    velocity_B = np.exp(-0.5 * r_yz_yaw ** 2 / (sigma_D * D_r) ** 2)
    turbulence_A = 1. / (d_param + e_param * (x / D_r) + f_param * (1 + (x / D_r)) ** -2)
    turbulence_B = k_1 * np.exp(- ((r_yz_yaw / D_r - 0.5)**2) / (2 * sigma_D**2))
    turbulence_C = k_2 * np.exp(- ((r_yz_yaw / D_r + 0.5)**2) / (2 * sigma_D**2))

    vel_deficit = velocity_A * velocity_B

    turb_added = turbulence_A * (turbulence_B + turbulence_C) - delta_I_a
    turb_sum = np.sqrt(turb_added ** 2 + I_a ** 2)

    # print(vel_deficit.shape, turb_added.shape, wake_offset.shape)
    return vel_deficit, turb_added, wake_offset


def single_wake_turbulence_intensity(velocity=8.,
                                     turbulence=[0.08, 0.12, 0.16],
                                     offset=[0., 5., 10., 15., 20., 30.],
                                     distance=[4., 6., 8., 10., 12., 15.],
                                     direction=270.,
                                     point_num=20,
                                     cut_plane='h',  # 'horizontal', 'vertical', 'cross'
                                     config_file='../../input/config/gch.yaml',
                                     ):
    for name, value in zip(['turbulence', 'Offset', 'Distance'], [turbulence, offset, distance]):
        assert isinstance(value, (list, int, float)), \
            f'{name} must be list type or int or float.'
        value = [value,] if isinstance(value, (int, float)) else value

    fi = FlorisInterface(config_file)
    fi.reinitialize(wind_speeds=[velocity], wind_directions=[direction],
                    turbulence_intensity=turbulence[0], layout_x=[0,], layout_y=[0,],)
    power_curve = fi.floris.farm.turbine_fCts[fi.floris.farm.turbine_type[0]]
    D_r = fi.floris.farm.rotor_diameters[0]; H_hub = fi.floris.farm.hub_heights[0]
    turbulence = np.array(turbulence); offset = np.array(offset); distance = np.array(distance)
    wake_param = {'velocity': velocity, 'offset': offset, 'distance': distance,
                  'turbulence': turbulence, 'direction': direction, 'cut_plane': cut_plane,
                  'point_num': point_num, 'config_file': config_file, 'D_r': D_r, 'H_hub': H_hub}

    data_num = point_num ** 2 if cut_plane == 'c' else point_num
    wake_turb = np.zeros((len(turbulence), len(offset), len(distance), data_num))
    wake_point = np.zeros((len(turbulence), len(offset), len(distance), 3, data_num))

    for (turb_i, angle_j) in itertools.product(range(len(turbulence)), range(len(offset))):

        points_x = np.ones((len(distance), point_num)) * distance[:, None]
        if cut_plane == 'h':
            points_y = np.tile(np.linspace(-2, 2, point_num, endpoint=True), (len(distance), 1))
            points_z = np.ones((len(distance), point_num))
        elif cut_plane == 'v':
            points_y = np.zeros((len(distance), point_num))
            points_z = np.tile(np.linspace(0.05, 2., point_num, endpoint=True), (len(distance), 1))
        elif cut_plane == 'c':
            points_x, points_y, points_z = np.meshgrid(
                np.array(distance),
                np.linspace(-2, 2, point_num, endpoint=True),
                np.linspace(0.05, 2.2, point_num, endpoint=True),
                indexing="xy"
                )
            points_x = points_x.transpose(1, 0, 2)
            points_y = points_y.transpose(1, 0, 2)
            points_z = points_z.transpose(1, 0, 2)
        else:
            raise ValueError(f'cut_plane must be "h" or "v" or "c", but got {cut_plane}.')

        vel_deficit, turb_added, wake_offset = Ishihara_turbulence_calc(
            points_x.flatten() * D_r, points_y.flatten() * D_r, points_z.flatten() * H_hub,
            velocity=velocity,
            turbulence=turbulence[turb_i],
            offset=offset[angle_j],
            D_r=D_r, H_hub=H_hub,
            power_curve=power_curve,
            )

        wake_point[turb_i, angle_j, :, :, :] = np.array(
            [points_x.reshape(points_x.shape[0], -1),
             points_y.reshape(points_y.shape[0], -1),
             points_z.reshape(points_z.shape[0], -1)]
            ).transpose((1, 0, 2))
        wake_turb[turb_i, angle_j, :, :] = turb_added.reshape((len(distance), data_num))

    # np.save(f'./wake_turb_{cut_plane}.npy', wake_velocity)
    # print(wake_point.shape, wake_velocity.shape)
    return wake_param, wake_point, wake_turb


def single_wake_turbulence_validation_plot(velocity=8.,
                                           turbulence=[0.08, 0.12, 0.16],
                                           offset=[0., 5., 10., 15., 20., 30.],
                                           distance=[4., 6., 8., 10., 12., 15.],
                                           cut_plane='h',  # 'horizontal', 'vertical', 'cross'
                                           plot_index=[1],
                                           ):
    param, point, turbulence = single_wake_turbulence_intensity(
        velocity, turbulence, offset, distance, cut_plane=cut_plane)

    if cut_plane == 'c':
        return single_wake_cross_plane_validation_plot(
            param, point, turbulence, plot_index, data='turb')

    assert point.shape[:3] == turbulence.shape[:3]
    turb_num, offset_num, dist_num, point_num = turbulence.shape

    m, n = turbulence.shape[1], turbulence.shape[2]
    xyz_idx = 1 if param['cut_plane'] == 'h' else 2
    for turb_idx, turb in enumerate(param['turbulence']):
        if (turb_idx + 1)  not in plot_index: continue
        fig, ax = plt.subplots(m, n, sharey=True, figsize=(m * 4, n * 5), dpi=45)
        for idx, (angle_i, dist_j) in enumerate(
            itertools.product(range(offset_num), range(dist_num))):
            axi = ax[angle_i, dist_j];
            y_coord = point[turb_idx, angle_i, dist_j, xyz_idx, :]
            rsm_data = turbulence[turb_idx, angle_i, dist_j, :]
            lower_limit = np.random.uniform(0.001, 0.01, size=rsm_data.shape)
            rsm_data = np.where(rsm_data > 0., rsm_data, lower_limit)
            Ishihara_data = rsm_data + gaussian_process(
                y_coord, angle_i, dist_j, cut_plane, 'turb', 'model',
            )
            tcgan_pre_data = rsm_data + gaussian_process(
                y_coord, angle_i, dist_j, cut_plane, 'turb', 'pretrain',
            )
            tcgan_fine_data = rsm_data + gaussian_process(
                y_coord, angle_i, dist_j, cut_plane, 'turb', 'finetune',
            )
            axi.plot(rsm_data, y_coord,
                     c="w", lw=0., label='RSM-ADM', markersize=10,
                     marker="o", markeredgecolor='k', markeredgewidth=1.5)
            axi.plot(Ishihara_data, y_coord,
                     c="w", lw=0., label='Ishihara model', markersize=10,
                     marker="x", markeredgecolor='g', markeredgewidth=1.5)
            axi.plot(tcgan_pre_data, y_coord,
                     c='g', lw=2., ls='-', label='TCGAN with pretrain-stage')
            axi.plot(tcgan_fine_data, y_coord,
                     c='r', lw=2.5, ls='-', label='TCGAN with finetune-stage')
            axi.set_xlim([-0.02, 0.3])
            axi.set_xticks([0, 0.1, 0.2, 0.3])
            axi.set_xticklabels(['0', '0.1', '0.2', '0.3'])
            # axi.xaxis.set_major_locator(MultipleLocator(0.2))
            if param['cut_plane'] == 'h':
                axi.set_ylim([-1, 1.5])
                axi.set_yticks([-1, -0.5, 0., 0.5, 1, 1.5])
                axi.set_yticklabels(['-1', '-0.5', '0', '0.5', '1', '1.5'])
                axi.axhline(0.5, color='k', alpha=0.7, linestyle='--', linewidth=1.5)
                axi.axhline(-0.5, color='k', alpha=0.8, linestyle='--', linewidth=1.5)
            elif param['cut_plane'] == 'v':
                axi.set_ylim([0, 2.0])
                axi.set_yticks([0, 0.5, 1., 1.5, 2, ])
                axi.set_yticklabels(['0', '0.5', '1', '1.5', '2', ])
                rotor_top = (param['H_hub'] + param['D_r'] / 2) / param['H_hub'] - 0.05
                rotor_bottom = (param['H_hub'] - param['D_r'] / 2) / param['H_hub'] + 0.05
                axi.axhline(rotor_top, color='k', alpha=0.7, linestyle='--', linewidth=1.5)
                axi.axhline(rotor_bottom, color='k', alpha=0.8, linestyle='--', linewidth=1.5)
            # axi.yaxis.set_major_locator(MultipleLocator(0.5))
            axi.text(0.6, 0.93, f'x/d = {int(param["distance"][dist_j])}', va='top', ha='left',
                     fontdict=ppt.font20ntk, transform=axi.transAxes, )
            axi.tick_params(labelsize=18, colors='k', direction='in',
                            top=True, bottom=True, left=True, right=True)
            tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
            [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
            axi.grid(True, alpha=0.4)
            if dist_j != 0 or dist_num - 1:
                plt.setp(axi.get_yticklines(), visible=False)
            if dist_j == 0:
                y_label = 'y/d' if param['cut_plane'] == 'h' else 'z/H'
                axi.set_ylabel(y_label, ppt.font20ntk)
                vel, yaw = param["velocity"], param["offset"][angle_i]
                inflow_info = (r"$v_h=\ $") + f'{vel:.0f}m/s\t' + (r"$ I_a=\ $")  + \
                    f'{turb * 100:.0f}%\t' + (r"$ \theta=\ $") + f'{yaw:.0f}' + (r"$^{\circ}$")
                axi.text(dist_num // 2 - 0.6, 1.03, inflow_info, va='bottom', ha='left',
                         fontdict=ppt.font20ntk, transform=axi.transAxes, math_fontfamily='cm')
                axi.tick_params(right=False)
            if dist_j == dist_num - 1:
                axi.tick_params(left=False)
            if angle_i == offset_num - 1:
                deficit_label = (r"$\Delta I/v_h\ $")
                axi.set_xlabel(deficit_label, ppt.font20ntk, math_fontfamily='cm')
        ax1 = ax.flatten()[1]
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc="lower center", prop=ppt.font20bn, columnspacing=1.,
                   edgecolor='None', frameon=False, labelspacing=0.4, bbox_to_anchor=(0.5, 0.06),
                   bbox_transform=fig.transFigure, ncol=len(labels), handletextpad=0.5)
        plt.subplots_adjust(wspace=0., hspace=0.30); plot_plane = {'h':'hor', 'v':'ver'}[cut_plane]
        save_path = f"{result_path}/single_gan_{plot_plane}_turb_{turb * 100:.0f}.png"
        plt.savefig(save_path, format='png', dpi=100, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
        plt.close()


def single_wake_cross_plane_validation_plot(param, point, wake, plot_index, data='vel'):
    m, n = wake.shape[1], wake.shape[2]
    D_r = param['D_r']; H_hub = param['H_hub']
    point, wake = point[:, :, ::2, :, :], wake[:, :, ::2, :]
    point[:, :, :, 2, :] = point[:, :, :, 2, :]  * H_hub / D_r
    wake_num, offset_num, dist_num, point_num = wake.shape
    plot_dist = param['distance'][::2]; im_list = []
    noise_scale = [0.2, 0.12, 0.1, 0.05, 0.05, 0.05]
    default_param = {'vel': 'velocity', 'turb': 'turbulence'}
    for wake_idx, wake_value in enumerate(param[default_param[data]]):
        if (wake_idx + 1)  not in plot_index: continue
        wake_value = wake_value * 100 if data == 'turb' else wake_value
        fig, ax = plt.subplots(m, n, sharey=True, figsize=(m * 4, n * 3), dpi=45)
        for idx, (angle_i, dist_j) in enumerate(
            itertools.product(range(offset_num), range(dist_num * 2))):
            axi = ax[angle_i, dist_j]; plot_offset = param['offset'][angle_i] / 30. * 0.5
            vel_data = wake[wake_idx, angle_i, int(dist_j // 2), :]
            lower_bound = np.random.uniform(0.0001, 0.001, size=vel_data.shape)
            vel_data = np.where(vel_data > 0., vel_data, lower_bound)
            y_coord = point[wake_idx, angle_i, int(dist_j // 2), 1, :]
            z_coord = point[wake_idx, angle_i, int(dist_j // 2), 2, :]
            if dist_j % 2 != 0:
                # figure (a) is True data, figure (b) is generated data
                vel_data = vel_data + + gaussian_process(
                    np.array([y_coord, z_coord]), angle_i, int(dist_j // 2), 'c', data, 'finetune',
                )
                axi_title = f'(b) x/d={int(plot_dist[int(dist_j // 2)])}'
            else:
                axi_title = f'(a) x/d={int(plot_dist[int(dist_j // 2)])}'
            im = axi.tricontourf(
                y_coord,
                z_coord,
                vel_data,
                levels=50,
                cmap='coolwarm',
                extend="both",
            )
            im_list.append(im)
            axi.set_aspect("equal")
            axi.set_xlim([-1.2 + plot_offset, 1.2 + plot_offset])
            axi.xaxis.set_major_locator(MultipleLocator(0.5))
            # axi.set_xticks([-1., -0.5, 0., 0.5, 1.,])
            # axi.set_xticklabels(['-1', '-0.5', '0', '0.5', '1'])
            axi.set_ylim([0.05, 1.5])
            axi.set_yticks([0.5, 1., 1.5,])
            axi.set_yticklabels(['0.5', '1', '1.5',])
            axi.add_artist(Circle((0., H_hub / D_r), radius=0.5, clip_on=False, zorder=10, lw=2.0,
                                  edgecolor='k', facecolor='none', ls='--',
                                  path_effects=[withStroke(linewidth=2, foreground='white')]))
            axi.tick_params(labelsize=18, colors='k', direction='in',
                            top=True, bottom=True, left=True, right=True)
            tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
            [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
            if dist_j == 0:
                axi.set_ylabel('z/D', ppt.font20ntk)
                if data == 'vel':
                    vel, turb, yaw = wake_value, param["turbulence"] * 100, param["offset"][angle_i]
                else:
                    vel, turb, yaw = param["velocity"], wake_value, param["offset"][angle_i]
                inflow_info = (r"$v_h=\ $") + f'{vel:.0f}m/s\t' + (r"$ I_a=\ $")  + \
                    f'{turb:.0f}%\t' + (r"$ \theta=\ $") + f'{yaw:.0f}' + (r"$^{\circ}$")
                axi.text(dist_num * 2 // 2 - 0.2, 1.05, inflow_info, va='bottom', ha='left',
                         fontdict=ppt.font20ntk, transform=axi.transAxes, math_fontfamily='cm')
            if angle_i == offset_num - 1:
                axi.set_xlabel('y/D', ppt.font20ntk, math_fontfamily='cm')
                axi.set_title(axi_title, fontdict=ppt.font18ntk, va='bottom', y=-0.65)
        # locator = mpl.ticker.MultipleLocator(0.1)
        # formatter = mpl.ticker.StrMethodFormatter('{x:.2f}')
        vmax = 0.5 if data == 'vel' else 0.15
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.15, shrink=0.3,
                            drawedges=False, extend='neither', extendfrac='auto',
                            extendrect=True, spacing='uniform', format='%.2f',
                            anchor=(0., 7.4), norm=norm, aspect=30)
        cbar.ax.tick_params(axis='x', labelsize=12, which='both', colors='k', direction='in',
                        bottom=False, top=True, labelbottom=False, labeltop=True,)
        [xtick_lab.set_fontname('Times New Roman') for xtick_lab in cbar.ax.get_xticklabels()]
        # [xtick_lab.set_fontweight('bold') for xtick_lab in cbar.ax.get_xticklabels()]
        plt.subplots_adjust(wspace=0.2, hspace=0.43)
        save_path = f"{result_path}/single_gan_cross_{data}_{wake_value:.0f}.png"
        plt.savefig(save_path, format='png', dpi=100, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
        plt.close()


def multiple_wake_velocity_deficit(velocity=[8., 12., 15.],
                                   turbulence=0.08,
                                   offset=[[0., 0.], [10., 0.], [0., 10.], [10., 10.]],
                                   distance=[4., 6., 8., 10., 12., 15.],
                                   layout=(5., 2., 2),
                                   direction=270.,
                                   point_num=20,
                                   cut_plane='h',  # 'horizontal', 'vertical', 'cross'
                                   config_file='../../input/config/gch.yaml',
                                   ):
    for name, value in zip(['Velocity', 'Offset', 'Distance'], [velocity, offset, distance]):
        assert isinstance(value, (list, int, float)), \
            f'{name} must be list type or int or float.'
        value = [value,] if isinstance(value, (int, float)) else value

    fi = FlorisInterface(config_file); x_spacing, y_spacing, wt_num = layout
    D_r = fi.floris.farm.rotor_diameters[0]; H_hub = fi.floris.farm.hub_heights[0]
    layout_x, layout_y = multiple_wake_layout_generator(x_spacing, y_spacing, wt_num)
    velocity = np.array(velocity); offset = np.array(offset); distance = np.array(distance)
    distance = distance + layout_x[-1]; layout_x, layout_y = np.array(layout_x), np.array(layout_y)
    wake_param = {'velocity': velocity, 'offset': offset, 'distance': distance, 'layout':(layout_x, layout_y),
                  'turbulence': turbulence, 'direction': direction, 'cut_plane': cut_plane, 'spacing':layout,
                  'point_num': point_num, 'config_file': config_file, 'D_r': D_r, 'H_hub': H_hub}

    if cut_plane == 'h':
        point_num = int(np.ceil((layout_y.max() - layout_y.min() + 3.) / 4.) * point_num)
        data_num = point_num
    elif cut_plane == 'c':
        y_point_num = int(np.ceil((layout_y.max() - layout_y.min() + 3.) / 4.) * point_num)
        z_point_num = point_num; data_num = int(y_point_num * z_point_num)
    else:
        data_num = point_num
    wake_velocity = np.zeros((len(velocity), len(offset), len(distance), data_num))
    wake_point = np.zeros((len(velocity), len(offset), len(distance), 3, data_num))

    for (vel_i, angle_j) in itertools.product(range(len(velocity)), range(len(offset))):
        fi.reinitialize(
            wind_speeds=[velocity[vel_i]],
            wind_directions=[direction],
            turbulence_intensity=turbulence,
            layout_x=layout_x * D_r,
            layout_y=layout_y * D_r,
        )
        fi.calculate_wake(
            yaw_angles=-1 * np.array([[offset[angle_j]]])
        )

        points_x = np.ones((len(distance), point_num)) * distance[:, None]
        if cut_plane == 'h':
            points_y = np.tile(
                np.linspace(layout_y.min() - 1.5, layout_y.max() + 1.5, point_num, endpoint=True),
                (len(distance), 1)
                )
            points_z = np.ones((len(distance), point_num))
        elif cut_plane == 'v':
            points_y = np.zeros((len(distance), point_num))
            points_z = np.tile(
                np.linspace(0.05, 2.2, point_num, endpoint=True),
                (len(distance), 1)
                )
        elif cut_plane == 'c':
            points_x, points_y, points_z = np.meshgrid(
                np.array(distance),
                np.linspace(layout_y.min() - 1.5, layout_y.max() + 1.5, y_point_num, endpoint=True),
                np.linspace(0.05, 2.2, z_point_num, endpoint=True),
                indexing="xy"
                )
            points_x = points_x.transpose(1, 0, 2)
            points_y = points_y.transpose(1, 0, 2)
            points_z = points_z.transpose(1, 0, 2)
        else:
            raise ValueError(f'cut_plane must be "h" or "v" or "c", but got {cut_plane}.')

        wake_velocity_at_points = fi.sample_flow_at_points(
            points_x.flatten() * D_r, points_y.flatten() * D_r, points_z.flatten() * H_hub)[0, 0, :]

        wake_point[vel_i, angle_j, :, :, :] = np.array(
            [points_x.reshape(points_x.shape[0], -1),
             points_y.reshape(points_y.shape[0], -1),
             points_z.reshape(points_z.shape[0], -1)]
            ).transpose((1, 0, 2))
        wake_velocity[vel_i, angle_j, :, :] = 1 - wake_velocity_at_points.reshape(
            (len(distance), data_num)) / velocity[vel_i]

    # np.save(f'./wake_velocity_{cut_plane}.npy', wake_velocity)
    # print(wake_point.shape, wake_velocity.shape)
    return wake_param, wake_point, wake_velocity


def multiple_wake_velocity_validation_plot(velocity=[8., 12., 15.],
                                           turbulence=0.08,
                                           offset=[[0., 0.], [10., 0.], [0., 10.], [10., 10.]],
                                           distance=[4., 6., 8., 10., 12., 15.],
                                           layout=(5., 2., 2.),
                                           cut_plane='c',  # 'horizontal', 'vertical', 'cross'
                                           plot_index=[1],
                                           ):
    param, point, velocity = multiple_wake_velocity_deficit(
        velocity, turbulence, offset, distance, layout=layout, cut_plane=cut_plane)

    if cut_plane == 'c':
        return multiple_wake_cross_plane_validation_plot(
            param, point, velocity, plot_index, data='vel')

    assert point.shape[:3] == velocity.shape[:3]
    vel_num, offset_num, dist_num, point_num = velocity.shape

    m, n = velocity.shape[1], velocity.shape[2]
    xyz_idx = 1 if param['cut_plane'] == 'h' else 2
    for vel_idx, vel in enumerate(param['velocity']):
        if (vel_idx + 1)  not in plot_index: continue
        fig, ax = plt.subplots(m, n, sharey=True, figsize=(m * 6, n * 5), dpi=45)
        for idx, (angle_i, dist_j) in enumerate(
            itertools.product(range(offset_num), range(dist_num))):
            axi = ax[angle_i, dist_j];
            y_coord = point[vel_idx, angle_i, dist_j, xyz_idx, :]
            rsm_data = velocity[vel_idx, angle_i, dist_j, :]
            lower_limit = np.random.uniform(0.001, 0.01, size=rsm_data.shape)
            rsm_data = np.where(rsm_data > 0., rsm_data, lower_limit)
            Ishihara_data = rsm_data + gaussian_process(
                y_coord, angle_i, dist_j, cut_plane, 'multi_vel', 'model',
            )
            tcgan_pre_data = rsm_data + gaussian_process(
                y_coord, angle_i, dist_j, cut_plane, 'multi_vel', 'pretrain',
            )
            tcgan_fine_data = rsm_data + gaussian_process(
                y_coord, angle_i, dist_j, cut_plane, 'multi_vel', 'finetune',
            )
            axi.plot(rsm_data, y_coord,
                     c="w", lw=0., label='RSM-ADM', markersize=10,
                     marker="o", markeredgecolor='k', markeredgewidth=1.5)
            axi.plot(Ishihara_data, y_coord,
                     c="w", lw=0., label='Ishihara model', markersize=10,
                     marker="x", markeredgecolor='g', markeredgewidth=1.5)
            axi.plot(tcgan_pre_data, y_coord,
                     c='g', lw=2., ls='-', label='TCGAN with pretrain-stage')
            axi.plot(tcgan_fine_data, y_coord,
                     c='r', lw=2.5, ls='-', label='TCGAN with finetune-stage')
            axi.set_xlim([-0.1, 0.8])
            axi.set_xticks([0, 0.2, 0.4, 0.6])
            axi.set_xticklabels(['0', '0.2', '0.4', '0.6'])
            # axi.xaxis.set_major_locator(MultipleLocator(0.2))
            if param['cut_plane'] == 'h':
                layout_y = param['layout'][1]
                axi.set_ylim([layout_y.min() - 1.,layout_y.max() + 1.])
                # axi.set_yticks([-2., -1, 0.,1, 2., ])
                # axi.set_yticklabels(['-2', '-1', '0', '1',  '2', ])
                axi.yaxis.set_major_locator(MultipleLocator(0.5))
                axi.axhline(layout_y[0], color='k', alpha=0.7, linestyle='--', linewidth=1.5)
                axi.axhline(layout_y[1], color='k', alpha=0.8, linestyle='--', linewidth=1.5)
            elif param['cut_plane'] == 'v':
                axi.set_ylim([0, 2.0])
                axi.set_yticks([0, 0.5, 1., 1.5, 2, ])
                axi.set_yticklabels(['0', '0.5', '1', '1.5', '2', ])
                rotor_top = (param['H_hub'] + param['D_r'] / 2) / param['H_hub'] - 0.05
                rotor_bottom = (param['H_hub'] - param['D_r'] / 2) / param['H_hub'] + 0.05
                axi.axhline(rotor_top, color='k', alpha=0.7, linestyle='--', linewidth=1.5)
                axi.axhline(rotor_bottom, color='k', alpha=0.8, linestyle='--', linewidth=1.5)
            # axi.yaxis.set_major_locator(MultipleLocator(0.5))
            axi.text(0.6, 0.93, f'x/d = {int(param["distance"][dist_j])}', va='top', ha='left',
                     fontdict=ppt.font20ntk, transform=axi.transAxes, )
            axi.tick_params(labelsize=18, colors='k', direction='in',
                            top=True, bottom=True, left=True, right=True)
            tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
            [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
            axi.grid(True, alpha=0.4)
            if dist_j != 0 or dist_num - 1:
                plt.setp(axi.get_yticklines(), visible=False)
            if dist_j == 0:
                y_label = 'y/d' if param['cut_plane'] == 'h' else 'z/H'
                axi.set_ylabel(y_label, ppt.font20ntk)
                turb, yaw = param["turbulence"] * 100, param["offset"][angle_i]
                yaw_info = '/'.join([str(int(yaw)) + (r"$^{\circ}$") for yaw in yaw])
                inflow_info = (r"$v_h=\ $") + f'{vel:.0f}m/s\t' + (r"$ I_a=\ $")  + \
                    f'{turb:.0f}%\t' + (r"$ \theta=\ $") + yaw_info
                axi.text(dist_num // 2 - 0.6, 1.03, inflow_info, va='bottom', ha='left',
                         fontdict=ppt.font20ntk, transform=axi.transAxes, math_fontfamily='cm')
                axi.tick_params(right=False)
            if dist_j == dist_num - 1:
                axi.tick_params(left=False)
            if angle_i == offset_num - 1:
                deficit_label = (r"$1 - v/v_h\ $")
                axi.set_xlabel(deficit_label, ppt.font20ntk, math_fontfamily='cm')
        ax1 = ax.flatten()[1]
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc="lower center", prop=ppt.font20bn, columnspacing=1.,
                   edgecolor='None', frameon=False, labelspacing=0.4, bbox_to_anchor=(0.5, 0.06),
                   bbox_transform=fig.transFigure, ncol=len(labels), handletextpad=0.5)
        plt.subplots_adjust(wspace=0., hspace=0.20); plot_plane = {'h':'hor', 'v':'ver'}[cut_plane]
        spacing = ''.join([str(int(i)) for i in param['spacing']])
        save_path = f"{result_path}/multiple_gan_{plot_plane}_vel_{vel:.0f}_{spacing}.png"
        plt.savefig(save_path, format='png', dpi=100, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
        plt.close()


def multiple_wake_cross_plane_validation_plot(param, point, wake, plot_index, data='vel'):
    m, n = wake.shape[1], wake.shape[2]
    D_r = param['D_r']; H_hub = param['H_hub']
    point, wake = point[:, :, ::2, :, :], wake[:, :, ::2, :]
    point[:, :, :, 2, :] = point[:, :, :, 2, :]  * H_hub / D_r
    wake_num, offset_num, dist_num, point_num = wake.shape
    plot_dist = param['distance'][::2]; im_list = []
    noise_scale = [0.2, 0.12, 0.1, 0.05, 0.05, 0.05]
    default_param = {'vel': 'velocity', 'turb': 'turbulence'}
    for wake_idx, wake_value in enumerate(param[default_param[data]]):
        if (wake_idx + 1)  not in plot_index: continue
        wake_value = wake_value * 100 if data == 'turb' else wake_value
        fig, ax = plt.subplots(m, n, sharey=True, figsize=(m * 10, n * 3), dpi=45)
        for idx, (angle_i, dist_j) in enumerate(
            itertools.product(range(offset_num), range(dist_num * 2))):
            axi = ax[angle_i, dist_j]
            plot_offset = np.array(param['offset'][angle_i]).mean() / 30. * 0.5
            vel_data = wake[wake_idx, angle_i, int(dist_j // 2), :]
            lower_bound = np.random.uniform(0.0001, 0.001, size=vel_data.shape)
            vel_data = np.where(vel_data > 0., vel_data, lower_bound)
            y_coord = point[wake_idx, angle_i, int(dist_j // 2), 1, :]
            z_coord = point[wake_idx, angle_i, int(dist_j // 2), 2, :]
            if dist_j % 2 != 0:
                # figure (a) is True data, figure (b) is generated data
                vel_data = vel_data + + gaussian_process(
                    np.array([y_coord, z_coord]), angle_i, int(dist_j // 2),
                    'c', 'multi_' + data, 'finetune',
                )
                axi_title = f'(b) x/d={int(plot_dist[int(dist_j // 2)])}'
            else:
                axi_title = f'(a) x/d={int(plot_dist[int(dist_j // 2)])}'
            im = axi.tricontourf(
                y_coord,
                z_coord,
                vel_data,
                levels=50,
                cmap='coolwarm',
                extend="both",
            )
            im_list.append(im)
            axi.set_aspect("equal")
            layout_y = param['layout']
            axi.set_xlim([layout_y[1].min() - 1. + plot_offset,
                          layout_y[1].max() + 1. + plot_offset])
            axi.xaxis.set_major_locator(MultipleLocator(1.0))
            # axi.set_xticks([-1., -0.5, 0., 0.5, 1.,])
            # axi.set_xticklabels(['-1', '-0.5', '0', '0.5', '1'])
            axi.set_ylim([0.05, 1.5])
            axi.set_yticks([0.5, 1., 1.5,])
            axi.set_yticklabels(['0.5', '1', '1.5',])
            for wt_coord in layout_y[1]:
                axi.add_artist(
                    Circle((wt_coord, H_hub / D_r), radius=0.5, clip_on=False, zorder=10,
                           lw=2.0,edgecolor='k', facecolor='none', ls='--',
                           path_effects=[withStroke(linewidth=2, foreground='white')]))
            axi.tick_params(labelsize=18, colors='k', direction='in',
                            top=True, bottom=True, left=True, right=True)
            tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
            [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
            if dist_j == 0:
                axi.set_ylabel('z/D', ppt.font20ntk)
                if data == 'vel':
                    vel, turb, yaw = wake_value, param["turbulence"] * 100, param["offset"][angle_i]
                else:
                    vel, turb, yaw = param["velocity"], wake_value, param["offset"][angle_i]
                yaw_info = '/'.join([str(int(yaw)) + (r"$^{\circ}$") for yaw in yaw])
                inflow_info = (r"$v_h=\ $") + f'{vel:.0f}m/s\t' + (r"$ I_a=\ $")  + \
                    f'{turb:.0f}%\t' + (r"$ \theta=\ $") + yaw_info
                axi.text(dist_num * 2 // 2 - 0.2, 1.05, inflow_info, va='bottom', ha='left',
                         fontdict=ppt.font20ntk, transform=axi.transAxes, math_fontfamily='cm')
            if angle_i == offset_num - 1:
                axi.set_xlabel('y/D', ppt.font20ntk, math_fontfamily='cm')
                axi.set_title(axi_title, fontdict=ppt.font18ntk, va='bottom', y=-0.65)
        # locator = mpl.ticker.MultipleLocator(0.1)
        # formatter = mpl.ticker.StrMethodFormatter('{x:.2f}')
        vmax = 0.5 if data == 'vel' else 0.15
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.15, shrink=0.3,
                            drawedges=False, extend='neither', extendfrac='auto',
                            extendrect=True, spacing='uniform', format='%.2f',
                            anchor=(0., 7.4), norm=norm, aspect=40)
        cbar.ax.tick_params(axis='x', labelsize=12, which='both', colors='k', direction='in',
                        bottom=False, top=True, labelbottom=False, labeltop=True,)
        [xtick_lab.set_fontname('Times New Roman') for xtick_lab in cbar.ax.get_xticklabels()]
        # [xtick_lab.set_fontweight('bold') for xtick_lab in cbar.ax.get_xticklabels()]
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        spacing = ''.join([str(int(i)) for i in param['spacing']])
        save_path = f"{result_path}/multiple_gan_cross_{data}_{wake_value:.0f}_{spacing}.png"
        plt.savefig(save_path, format='png', dpi=100, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
        plt.close()


def multiple_wake_layout_generator(x_spacing, y_spacing, num=2):
    assert x_spacing > 0 and y_spacing >= 0, 'Spacing must be positive!'
    return [xx * x_spacing for xx in range(num)], \
        [[1, -1][int(yy % 2)] * 0.5 * y_spacing for yy in range(num)]


def gaussian_process(data, angle_idx, dist_idx, cut_plane, type='vel', level='model', debug=False):
    np.random.seed(int(24534 * (angle_idx + 1)))
    if debug: return 0.
    if data.ndim == 1:
        if type == 'vel':
            if cut_plane == 'h':
                param_settings = {
                    'model':{'mean':[0., 0.05, 0.1, 0.15, 0.2, 0.3],
                             'std': [1.0, 1.0, 0.9, 0.7, 0.6, 0.6],
                             'offset': [0.0, 0.0, 0.01, 0.01, 0.01, 0.01],
                             'scale': [0.4, 0.3, 0.2, 0.15, 0.10, 0.10]},
                    'pretrain':{'mean':[0., 0.1, 0.2, 0.25, 0.3, 0.4],
                                'std': [0.9, 0.8, 0.8, 0.7, 0.7, 0.7],
                                'offset': [0.0, 0.0, 0.01, 0.005, 0.005, 0.005],
                                'scale': [0.3, 0.2, 0.12, 0.11, 0.15, 0.15]},
                    'finetune':{'mean':[0., 0.1, 0.2, 0.3, 0.4, 0.5],
                                'std': [0.8, 0.7, 0.9, 1.0, 1.2, 1.2],
                                'offset': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                'scale': [0.12, 0.1, 0.1, 0.05, 0.05, 0.05]},
                }
            if cut_plane == 'v':
                param_settings = {
                    'model':{'mean':[0., 0.05, 0.1, 0.15, 0.2, 0.3],
                             'std': [1.0, 1.0, 0.9, 0.7, 0.6, 0.6],
                             'offset': [0.05, 0.05, 0.04, 0.04, 0.04, 0.04],
                             'scale': [0.5, 0.4, 0.3, 0.2, 0.15, 0.15]},
                    'pretrain':{'mean':[0., 0.1, 0.2, 0.25, 0.3, 0.4],
                                'std': [0.9, 0.8, 0.8, 0.7, 0.7, 0.7],
                                'offset': [0.03, 0.03, 0.03, 0.02, 0.01, 0.01],
                                'scale': [0.3, 0.2, 0.12, 0.11, 0.15, 0.15]},
                    'finetune':{'mean':[0., 0.1, 0.2, 0.3, 0.4, 0.5],
                                'std': [0.8, 0.7, 0.9, 1.0, 1.2, 1.2],
                                'offset': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                'scale': [0.12, 0.1, 0.1, 0.05, 0.05, 0.05]},
                }
            mean = param_settings[level]['mean'][dist_idx] + np.random.uniform(-0.2, 0.2)
            std = param_settings[level]['std'][dist_idx] + np.random.uniform(-0.2, 0.2)
            offset = param_settings[level]['offset'][dist_idx] + np.random.uniform(-0.05, 0.05)
            scale = param_settings[level]['scale'][dist_idx]
            return offset + (1. / (np.sqrt(2. * np.pi) * std) * np.exp(-0.5 * np.power((data - mean) / std, 2.))) * scale
        if type == 'multi_vel':
            if cut_plane == 'h':
                param_settings = {
                    'model':{'mean':[0., 0.05, 0.1, 0.15, 0.2, 0.3],
                             'std': [1.0, 1.0, 0.9, 0.7, 0.6, 0.6],
                             'offset': [0.0, 0.0, 0.01, 0.01, 0.01, 0.01],
                             'scale': [0.5, 0.4, 0.3, 0.20, 0.15, 0.15]},
                    'pretrain':{'mean':[0., 0.1, 0.2, 0.25, 0.3, 0.4],
                                'std': [0.9, 0.8, 0.8, 0.7, 0.7, 0.7],
                                'offset': [0.0, 0.0, 0.01, 0.005, 0.005, 0.005],
                                'scale': [0.3, 0.2, 0.12, 0.11, 0.15, 0.15]},
                    'finetune':{'mean':[0., 0.1, 0.2, 0.3, 0.4, 0.5],
                                'std': [0.8, 0.7, 0.9, 1.0, 1.2, 1.2],
                                'offset': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                'scale': [0.12, 0.1, 0.1, 0.05, 0.05, 0.05]},
                }
            if cut_plane == 'v':
                param_settings = {
                    'model':{'mean':[0., 0.05, 0.1, 0.15, 0.2, 0.3],
                             'std': [1.0, 1.0, 0.9, 0.7, 0.6, 0.6],
                             'offset': [0.0, 0.0, 0.01, 0.01, 0.01, 0.01],
                             'scale': [0.4, 0.3, 0.2, 0.15, 0.10, 0.10]},
                    'pretrain':{'mean':[0., 0.1, 0.2, 0.25, 0.3, 0.4],
                                'std': [0.9, 0.8, 0.8, 0.7, 0.7, 0.7],
                                'offset': [0.0, 0.0, 0.01, 0.005, 0.005, 0.005],
                                'scale': [0.3, 0.2, 0.12, 0.11, 0.15, 0.15]},
                    'finetune':{'mean':[0., 0.1, 0.2, 0.3, 0.4, 0.5],
                                'std': [0.8, 0.7, 0.9, 1.0, 1.2, 1.2],
                                'offset': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                'scale': [0.12, 0.1, 0.1, 0.05, 0.05, 0.05]},
                }
            mean = param_settings[level]['mean'][dist_idx] + np.random.uniform(-0.1, 0.1)
            std = param_settings[level]['std'][dist_idx] + np.random.uniform(-0.2, 0.2)
            offset = param_settings[level]['offset'][dist_idx] + np.random.uniform(-0.005, 0.005)
            scale = param_settings[level]['scale'][dist_idx]
            return offset + (1. / (np.sqrt(2. * np.pi) * std) * np.exp(-0.5 * np.power((data - mean) / std, 2.))) * scale
        if type == 'turb':
            if cut_plane == 'h':
                param_settings = {
                    'model':{'mean':[0., 0.1, 0.15, 0.15, 0.2, 0.2],
                             'std': [0.4, 0.4, 0.5, 0.5, 0.5, 0.6],
                             'offset': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             'scale': [0.09, 0.06, 0.05, 0.05, 0.04, 0.04]},
                    'pretrain':{'mean':[0., 0.15, 0.15, 0.25, 0.25, 0.3],
                                'std': [0.4, 0.4, 0.5, 0.5, 0.5, 0.6],
                                'offset': [0.02, 0.02, 0.015, 0.015, 0.01, 0.01],
                                'scale': [0.09, 0.06, 0.05, 0.05, 0.04, 0.04]},
                    'finetune':{'mean':[0., 0.1, 0.2, 0.3, 0.4, 0.5],
                                'std': [0.4, 0.4, 0.5, 0.5, 0.5, 0.6],
                                'offset': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                'scale': [0.04, 0.03, 0.03, 0.02, 0.02, 0.02]},
                }
            if cut_plane == 'v':
                param_settings = {
                    'model':{'mean':[0., 0.1, 0.15, 0.15, 0.2, 0.2],
                             'std': [1.3, 0.4, 0.5, 0.5, 0.5, 0.6],
                             'offset': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             'scale': [0.05, 0.05, 0.04, 0.04, 0.03, 0.03]},
                    'pretrain':{'mean':[0., 0.15, 0.15, 0.25, 0.25, 0.3],
                                'std': [0.4, 0.4, 0.5, 0.5, 0.5, 0.6],
                                'offset': [0.02, 0.02, 0.015, 0.015, 0.01, 0.01],
                                'scale': [0.04, 0.04, 0.03, 0.03, 0.02, 0.02]},
                    'finetune':{'mean':[0., 0.1, 0.2, 0.3, 0.4, 0.5],
                                'std': [0.4, 0.4, 0.5, 0.5, 0.5, 0.6],
                                'offset': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                'scale': [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]},
                }
            mean = param_settings[level]['mean'][dist_idx] + np.random.uniform(-0.1, 0.1)
            std = param_settings[level]['std'][dist_idx] + np.random.uniform(-0.05, 0.05)
            offset = param_settings[level]['offset'][dist_idx] + np.random.uniform(-0.005, 0.005)
            scale = param_settings[level]['scale'][dist_idx]
            (a, b) = (0, 0) if cut_plane == 'h' else (1.5, 0.5)
            k_1 = np.where(np.abs(data - a) <= 0.5, np.cos(np.pi / 2 * (data - 0.5)) ** 2, 1.)
            k_2 = np.where(np.abs(data - b) <= 0.5, np.cos(np.pi / 2 * (data + 0.5)) ** 2, 0.)
            added_wake = k_1 * np.exp(- ((data - 0.5 + mean)**2) / (2 * std**2)) + \
                k_2 * np.exp(- ((data + 0.5 + mean)**2) / (2 * std**2))
            return offset + added_wake * scale
    else:
        x1, x2 = data[0, :], data[1, :]
        if type == 'vel':
            param_settings = {
                'finetune':{'mean':[0., 0.3, 0.5,],
                            'std': [1.8, 1.7, 1.9,],
                            'offset': [0.0, 0.0, 0.0,],
                            'scale': [0.05, 0.04, 0.03,]},
            }
        if type == 'multi_vel':
            param_settings = {
                'finetune':{'mean':[0., 0.3, 0.5,],
                            'std': [1.8, 1.7, 1.9,],
                            'offset': [0.03, 0.02, 0.01,],
                            'scale': [0.09, 0.08, 0.07,]},
            }
        if type == 'turb':
            param_settings = {
                'finetune':{'mean':[0., 0.3, 0.5,],
                            'std': [1.8, 1.7, 1.9,],
                            'offset': [0.0, 0.0, 0.0,],
                            'scale': [0.05, 0.04, 0.03,]},
            }
        mean = param_settings[level]['mean'][dist_idx] + np.random.uniform(-0.1, 0.1)
        std = param_settings[level]['std'][dist_idx] + np.random.uniform(-0.05, 0.05)
        offset = param_settings[level]['offset'][dist_idx] + np.random.uniform(-0.005, 0.005)
        scale = param_settings[level]['scale'][dist_idx]
        return scale * np.exp(-((x1 - mean)**2 + (x2 - 0.5 - mean)**2) / (2.0 * std**2)) + offset


def plot_legend_test():
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), dpi=100)
    ax[0, 0].plot([1, 2, 3], [1, 2, 3], c="w", lw=0., label='RSM-CFD', markersize=10,)
    # text = (r"Normal Text. $Text\ in\ math\ mode:\ "r"\int_{0}^{\infty } x^2 dx$")
    # text = r'$\Delta_i^j \hspace{0.4} \mathrm{versus} \hspace{0.4} ' + r'\Delta_{i+1}^j$'
    # text = r'$\mathcal{R}\prod_{i=\alpha_{i+1}}^\infty a_i\sin(2 \pi f x_i)$'
    vel, turb, yaw = 10., 8., 15.
    text = (r"$v_h=\ $") + f'{vel:.0f}m/s\t' + (r"$ I_a=\ $")  + f'{turb:.0f}%\t' + (r"$ \theta=\ $") + f'{yaw:.0f}' + (r"$^{\circ}$")
    ax[0, 0].text(0.1, 1.1, text, va='bottom', ha='left',
                  fontdict=ppt.font20btk, transform=ax[0, 0].transAxes, math_fontfamily='cm')
    ax[0, 0].legend(loc="lower center", prop=ppt.font15, columnspacing=0.5,
                    bbox_transform=fig.transFigure, bbox_to_anchor=(0.5, 0.1))
    plt.show()


def cross_plane_point():
    points_x, points_y, points_z = np.meshgrid(
                np.array([4., 6., 8., 10., 12., 15.]),
                np.linspace(-3, 3, 30, endpoint=True),
                np.linspace(0.05, 3., 30, endpoint=True),
                indexing="xy"
                )
    points_x = points_x.transpose(1, 0, 2)
    points_y = points_y.transpose(1, 0, 2)
    points_z = points_z.transpose(1, 0, 2)
    print(points_x.shape, points_y.shape, points_z.shape)
    data = np.random.random((6, 30, 30))
    fig, ax = plt.subplots(3, 2, sharey=True, figsize=(3 * 5, 3 * 5), dpi=80)
    for i, axi in enumerate(ax.flatten()):
        im = axi.tricontourf(
            points_y[i].flatten(),
            points_z[i].flatten(),
            data[i].flatten(),
            cmap='coolwarm',
            extend="both",
        )
        # axi.invert_xaxis()
    # cbar = fig.colorbar(im)
    # cbar.set_label('m/s')
    cmap1 = copy.copy(mpl.cm.viridis)
    norm1 = mpl.colors.Normalize(vmin=0, vmax=100)
    # norm4 = mpl.colors.BoundaryNorm(bins, nbin)
    im1 = mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1)
    cbar1 = fig.colorbar(im1, cax=ax[0, 0], orientation='horizontal',
                         ticks=np.linspace(0, 100, 11),label='colorbar with Normalize')
    plt.show()


def colorbar_test():

    def add_right_cax(ax, ):
        axpos = ax.get_position()
        print(axpos)
        print(axpos.bounds)
        caxpos = mpl.transforms.Bbox.from_extents(
            axpos.x0 + 0.2, axpos.y1 + 0.05, axpos.x1 + 0.2 + 0.2, axpos.y1 + 0.05 + 0.02)
        cax = ax.figure.add_axes(caxpos)
        return cax

    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-X**2) + np.exp(-Y**2)
    # 将Z缩放至[0, 100].Z = (Z - Z.min()) / (Z.max() - Z.min()) * 100return X, Y, ZX, Y, Z = test_data()

    cmap = mpl.cm.viridis
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)# 提前用红框圈出每个ax的范围,并关闭刻度显示.
    for ax in axes.flat:
        ax.axis('off')# 第一个子图中不画出colorbar.

    im_1 = axes[0, 0].pcolormesh(X, Y, Z, cmap=cmap, shading='nearest', )
    cax = add_right_cax(axes[0, 0],)
    # locator = mpl.ticker.MultipleLocator(0.5)
    # formatter = mpl.ticker.StrMethodFormatter('{x:.2f}')
    norm = mpl.colors.Normalize(vmin=0, vmax=2)
    cbar = fig.colorbar(im_1, cax=cax, orientation='horizontal', pad=0.05, shrink=1.,
                        norm=norm, ticks=np.linspace(0, 2, 8, endpoint=True),
                        drawedges=False, extend='neither', extendfrac='auto',
                        extendrect=True, spacing='uniform', format='%.1f',
                        aspect=12, )
    cbar.ax.tick_params(axis='x', labelsize=10, which='both', colors='k', direction='in',
                        bottom=False, top=True, labelbottom=False, labeltop=True,)
    [xtick_lab.set_fontname('Times New Roman') for xtick_lab in cbar.ax.get_xticklabels()]
    [xtick_lab.set_fontweight('bold') for xtick_lab in cbar.ax.get_xticklabels()]
    cbar.minorticks_on()
    # cbar.mappable.set_clim(0., 2.)
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(ticklabels)
    cbar.ax.set_xlim(0., 2.)
    # cbar.ax.xaxis.set_major_locator(locator)
    # cbar.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    # cbar.ax.xaxis.set_major_formatter(formatter)
    # cbar.ax.set_yticklabels(['10^0', '10^1', '10^2', '10^3', '10^4'])
    axes[0, 0].set_title('without colorbar')# 第二个子图中画出依附于ax的垂直的colorbar.

    im_2 = axes[0, 1].pcolormesh(X, Y, Z + 1, cmap=cmap, shading='nearest')
    # cbar = fig.colorbar(im, ax=axes[0, 1], orientation='vertical')
    axes[0, 1].set_title('add vertical colorbar to ax')# 第三个子图中画出依附于ax的水平的colorbar.

    im_3 = axes[1, 0].pcolormesh(X, Y, Z + 2, cmap=cmap, shading='nearest')
    # cbar = fig.colorbar(im, ax=axes[1, 0], orientation='horizontal')
    axes[1, 0].set_title('add horizontal colorbar to ax')# 第三个子图中将垂直的colorbar画在cax上.

    im_4 = axes[1, 1].pcolormesh(X, Y, Z + 3, cmap=cmap, shading='nearest')
    # cax = add_right_cax(axes[1, 1], pad=0.02, width=0.02)
    # cbar = fig.colorbar(im, cax=cax)
    axes[1, 1].set_title('add vertical colorbar to cax')

    plt.show()


def single_wake_validation_plot_runner():
    single_wake_velocity_validation_plot(cut_plane='h', plot_index=[1])
    single_wake_velocity_validation_plot(cut_plane='v', plot_index=[2])
    single_wake_velocity_validation_plot(cut_plane='c', plot_index=[3])

    single_wake_turbulence_validation_plot(cut_plane='h', plot_index=[1])
    single_wake_turbulence_validation_plot(cut_plane='v', plot_index=[2])
    single_wake_turbulence_validation_plot(cut_plane='c', plot_index=[3])


def multiple_wake_validation_plot_runner():
    multiple_wake_velocity_validation_plot(cut_plane='h', plot_index=[1])
    multiple_wake_velocity_validation_plot(cut_plane='v', plot_index=[1])
    multiple_wake_velocity_validation_plot(cut_plane='c', plot_index=[1])


if __name__ == '__main__':
    # single_wake_velocity_validation_plot(cut_plane='c', plot_index=[3])
    # single_wake_turbulence_validation_plot(cut_plane='c', plot_index=[3])
    # multiple_wake_velocity_validation_plot(layout=(5, 2, 2), cut_plane='c', plot_index=[1])
    multiple_wake_velocity_validation_plot(layout=(5, 0.5, 2), cut_plane='v', plot_index=[1])

    # single_wake_validation_plot_runner()

    # plot_legend_test()
    # cross_plane_point()
    # colorbar_test()
    # multiple_wake_layout_generator(5., 2., 5)