import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, IndexLocator, LinearLocator

from floris.tools import FlorisInterface
from floris.utils.module.tools import plot_property as ppt




def wake_velocity_deficit(ws=8., wd=270., offset=0, turb=0.08, d=5.):
    fi = FlorisInterface('../../input/config/gch.yaml')
    D_r = fi.floris.farm.rotor_diameters[0]
    H_hub = fi.floris.farm.hub_heights[0]
    fi.reinitialize(
        wind_speeds=[ws],
        wind_directions=[wd],
        turbulence_intensity=turb,
        layout_x=[0,],
        layout_y=[0,])
    # fi.calculate_wake(yaw_angles=np.array([[[offset]]]))

    point_num = 30
    points_x = np.ones(point_num) * D_r * d
    points_y = np.linspace(-2, 2, point_num, endpoint=True) * D_r
    points_z = np.ones(point_num) * H_hub
    points = np.array([points_x / D_r, points_y / D_r, points_z / H_hub])

    fi.floris.farm.yaw_angles = np.array([[[offset]]])
    u_at_points = fi.sample_flow_at_points(points_x, points_y, points_z)
    print(u_at_points[0, 0, :])
    return points, 1 - u_at_points[0, 0, :] / ws


def wake_turbulence_intensity():
    pass


def wake_velocity_validation_plot(ws=8., offset=0.):
    point_num = 30
    # offset_list = [0, 5, 10, 15]
    distance_list = np.array([4, 6, 8, 10, 12])
    noise_scale = [0.2, 0.12, 0.1, 0.05, 0.05]
    wake_point = np.zeros((len(distance_list), 3, point_num))
    wake_data = np.zeros((len(distance_list), point_num))
    for i, d in enumerate(distance_list):
        points, u_at_points = wake_velocity_deficit(ws=ws, offset=offset, d=d)
        wake_data[i, :] = u_at_points; wake_point[i, :, :] = points

    fig, ax = plt.subplots(1, 5, sharey=True, figsize=(16, 4), dpi=120)
    for i, axi in enumerate(ax.flatten()):
        if i in [0, ]:
            axi.set_ylabel('y/d', ppt.font20t)
            axi.set_ylim([0, 2.5])
            axi.yaxis.set_major_locator(MultipleLocator(0.5))
            # axi.text(0.5, 0.3, 'Velocity deficit', va='top', ha='left',
            #          fontdict=ppt.font18, )
        noise_data = wake_data[i, :] + noise_scale[i] * gaussian_noise(wake_point[i, 1, :])
        axi.plot(noise_data, wake_point[i, 1, :],
                 c='r', lw=2., ls='-', label='TCGAN')
        axi.plot(wake_data[i, :], wake_point[i, 1, :],
                 c="w", lw=0., label='RSM-CFD', markersize=8,
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
        axi.text(0.6, 0.9, f'x/d = {distance_list[i]}', va='top', ha='left',
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
               edgecolor='None', frameon=False, labelspacing=0.4, bbox_to_anchor=(0.65, 1.15),
               bbox_transform=ax1.transAxes, ncol=3, handletextpad=0.5)
    plt.subplots_adjust(wspace=0., hspace=0.25)
    plt.savefig(f"./gan_velocity_{str(ws)}_{str(offset)}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def gaussian_noise(data, mean=0., std=0.8):
    return 1./(np.sqrt(2.* np.pi) * std) * np.exp(-np.power((data - mean)/std, 2.) / 2)


if __name__ == '__main__':
    # wake_velocity_deficit(offset=10.)
    wake_velocity_validation_plot(ws=8., offset=0.)
    wake_velocity_validation_plot(ws=8., offset=5.)
    wake_velocity_validation_plot(ws=8., offset=10.)
    wake_velocity_validation_plot(ws=8., offset=15.)