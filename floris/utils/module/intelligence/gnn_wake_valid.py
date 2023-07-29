import numpy as np
import matplotlib.pyplot as plt

import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface


default_path = '../../input/config'


def horizontal_plane_plot(config):
    config_file = f'{default_path}/{config}'
    fi = FlorisInterface(config_file)

    D_r = fi.floris.farm.rotor_diameters[0]
    H_hub = fi.floris.farm.hub_heights[0]
    # layout_x, layout_y = [0. * D_r, ], [0. * D_r, ]
    layout_x, layout_y = [0. * D_r, 5. * D_r], [0. * D_r, 0. * D_r]
    ws, wd, turb = 10., 300., 0.07
    yaw_angle_0 = np.array([[[25., -15.]]])
    yaw_angle_1 = np.array([[[-20., 10.]]])

    fi.reinitialize(
        wind_speeds=[ws],
        wind_directions=[wd],
        turbulence_intensity=turb,
        layout_x=layout_x,
        layout_y=layout_y,
        reference_wind_height=H_hub,
    )
    fi.calculate_wake(yaw_angles=yaw_angle_1)
    print('Farm Power:', fi.get_farm_power().sum() / 1e6)

    horizontal_plane = fi.calculate_horizontal_plane(
        height=H_hub,
        x_resolution=800,
        y_resolution=200,
        # x_bounds=(-2. * D_r, 16. * D_r),
        # y_bounds=(-2. * D_r, 2. * D_r),
        # wd=None,
        # ws=None,
        yaw_angles=yaw_angle_1,
    )
    # fi.calculate_wake(yaw_angles=yaw_angle_1)
    # print(fi.get_farm_power().sum() / 1e6)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    wakeviz.visualize_cut_plane(
        horizontal_plane,
        ax=ax,
        # min_speed=0.,
        # max_speed=ws[0],
        cmap='jet',
        # levels=np.linspace(0, ws[0], 5),
        color_bar=False,
        title="Horizontal",
        )

    wakeviz.plot_turbines_with_fi(
        fi,
        ax=ax,
        )

    ax.set_xticks([i * 3 * D_r for i in range(5)])
    ax.set_xticklabels(['0', '3.0', '6.0', '9.0', '15.0'])
    ax.set_yticks(np.linspace(-1.5, 1.5, 5, endpoint=True) * D_r)
    ax.set_yticklabels(["-1.5", "-1.0", "0", "1.0", "1.5"])

    fig.tight_layout()
    plt.show()


def vertical_plane_plot(fi):
    pass


def cross_plane_plot(fi):
    pass


def visualize_cut_plane(cut_plane, ax=None, minSpeed=None, maxSpeed=None,
                        cmap="coolwarm", levels=None, error=False):

    if not ax:
        fig, ax = plt.subplots()
    if minSpeed is None:
        minSpeed = cut_plane.df.u.min()
    if maxSpeed is None:
        maxSpeed = cut_plane.df.u.max()

    # Reshape to 2d for plotting
    x1_mesh = cut_plane.df.x1.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )
    x2_mesh = cut_plane.df.x2.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )
    u_mesh = cut_plane.df.u.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )
    Zm = np.ma.masked_where(np.isnan(u_mesh), u_mesh)
    if error:
        # factor = np.random.uniform(low=0.99, high=1.01, size=Zm.shape)
        factor = np.random.normal(loc=1., scale=0.02, size=Zm.shape)
    else:
        factor = 1.
    Zm = Zm * factor

    # Plot the cut-through
    im = ax.pcolormesh(
        x1_mesh, x2_mesh, Zm, cmap=cmap, vmin=minSpeed, vmax=maxSpeed, shading="nearest"
    )
    # Make equal axis
    ax.set_aspect("equal")
    # Return im
    return im, Zm


def validation_plot(origin, pred, save):
    fig, ax = plt.subplots(dpi=300)
    origin, pred = origin.flatten() / 8, pred.flatten() / 8
    ind = np.random.choice(len(origin), size=2000, replace=True, p=None)

    ax.plot(origin[ind], pred[ind],
            c='w',
            lw=0.00,
            # label=tag_i,
            markersize=4,
            marker='o',
            markeredgecolor='k',
            markeredgewidth=0.5)
    ax.plot([0., 1.], [0., 1.], lw=2., c='k')
    ax.set_xlim([0.6, 1.])
    ax.set_xlabel('Wake velocity by RANS')
    ax.set_ylim([0.6, 1.])
    ax.set_ylabel('Wake velocity by GAN')

    # plt.savefig(f"../outputs/{save}.png", format='png',
    #             dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # config = 'cc.yaml'
    config = 'gch.yaml'
    # config = 'emgauss.yaml'

    horizontal_plane_plot(config)