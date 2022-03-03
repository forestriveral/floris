# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np
from floris.simulation import turbine
import matplotlib.pyplot as plt

import floris.tools as wfct



config = "../inputs/single_wake.json"
fi = wfct.floris_interface.FlorisInterface(config)


def hrizontal_plane_plot(fi):
    D = fi.floris.farm.turbine_map.turbines[0].rotor_diameter
    wd, yaw_angles = [270.], [0.]
    fi.reinitialize_flow_field(wind_direction=wd, layout_array=([0], [0]))
    fi.calculate_wake(yaw_angles=yaw_angles)

    # Get horizontal plane at default height (hub-height)
    hor_plane = fi.get_hor_plane(x_resolution=200,
                                 y_resolution=200,
                                 x_bounds=(-2. * D, D * 16.),
                                 y_bounds=(-2. * D, 2. * D),
                                 )

    # Plot and show
    fig, ax = plt.subplots(dpi=300)
    visualize_cut_plane(hor_plane, ax=ax, cmap="coolwarm",)
    wfct.visualization.plot_turbines(ax, fi.layout_x, fi.layout_y,
                                     fi.get_yaw_angles(), D * 1.4,
                                     wind_direction=wd)
    ax.set_xticks([i * 3 * D for i in range(5)])
    ax.set_xticklabels(['0', '3.0', '6.0', '9.0', '15.0'])
    ax.set_yticks([-1.0 * D, 0.0, 1.0 * D])
    ax.set_yticklabels(["-1.0", "0", "1.0"])
    # plt.savefig(f"../../floris/utils/outputs/wf-2-3d.png", format='png',
    #             dpi=300, bbox_inches='tight')
    plt.show()


def cross_plane_plot(fi, dist, yaw=0, error=False):
    D = fi.floris.farm.turbine_map.turbines[0].rotor_diameter
    wd, yaw_angles = [270.], [yaw]
    fi.reinitialize_flow_field(wind_direction=wd, layout_array=([0], [0]))
    fi.calculate_wake(yaw_angles=yaw_angles)

    # Grab some cross planes
    distance = dist * D
    cut_plane = fi.get_cross_plane(distance,
                                   y_resolution=200,
                                   z_resolution=200,
                                   y_bounds=(-150., 150.),
                                #    z_bounds=(0., 180.),
                                   )

    fig, ax = plt.subplots(dpi=300)
    _, zm = visualize_cut_plane(cut_plane, ax=ax, cmap="coolwarm", error=error)
    ax.set_xticks([-126., -63., 0., 63., 126.])
    ax.set_xticklabels(['-1.0', '-0.5', '0', '0.5', '1.0',])
    ax.set_yticks([45., 90., 135.])
    ax.set_yticklabels(['0.5', '1.0', '1.5',])
    error_flag = 'e' if error else 'o'
    plt.savefig(f"../outputs/c-{int(yaw)}-{int(dist)}d-{error_flag}.png", format='png',
                dpi=300, bbox_inches='tight')
    # plt.show()

    return zm


def vertical_plane_plot(fi):
    D = fi.floris.farm.turbine_map.turbines[0].rotor_diameter
    wd, yaw_angles = [270.], [25.]
    fi.reinitialize_flow_field(wind_direction=wd, layout_array=([0], [0]))
    fi.calculate_wake(yaw_angles=yaw_angles)

    # Grab some cross planes
    # distance = 3 * D
    cut_plane = fi.get_y_plane(0.0,
                            #    x_resolution=200,
                            #    z_resolution=200,
                            #    x_bounds=(-2. * D, D * 16.),
                            #    z_bounds=(0., 250),
                               )

    fig, ax = plt.subplots(dpi=300)
    wfct.visualization.visualize_cut_plane(cut_plane, ax=ax, cmap="coolwarm",)
    # ax.set_xticks([i * 3 * D for i in range(5)])
    # ax.set_xticklabels(['0', '3.0', '6.0', '9.0', '15.0'])
    # ax.set_yticks([45., 90., 135.])
    # ax.set_yticklabels(['0.5', '1.0', '1.5',])
    # plt.savefig(f"../../floris/utils/outputs/wf-2-3d.png", format='png',
    #             dpi=300, bbox_inches='tight')
    plt.show()


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

    plt.savefig(f"../outputs/{save}.png", format='png',
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # hrizontal_plane_plot(fi)

    dist, yaw = -1., 0.
    ozm = cross_plane_plot(fi, dist, yaw, error=False)
    # ezm = cross_plane_plot(fi, dist, yaw, error=True)

    # vertical_plane_plot(fi)

    # validation_plot(ozm, ezm, save=f'error-{int(yaw)}-{int(dist)}d')
