import numpy as np
import matplotlib.pyplot as plt


from module import (
    Farm,
    TurbineGrid,
    PlanarGrid,
    FlowField,
)


def visualize_cut_plane(
    cut_plane,
    ax=None,
    vel_component='u',
    min_speed=None,
    max_speed=None,
    cmap="coolwarm",
    levels=None,
    clevels=None,
    color_bar=False,
    title="",
    **kwargs
):
    """
    Generate pseudocolor mesh plot of the cut_plane.

    Args:
        cut_plane (:py:class:`~.tools.cut_plane.CutPlane`): 2D
            plane through wind plant.
        ax (:py:class:`matplotlib.pyplot.axes`, optional): Figure axes. Defaults
            to None.
        vel_component (str, optional): The velocity component that the cut plane is
            perpendicular to.
        min_speed (float, optional): Minimum value of wind speed for
            contours. Defaults to None.
        max_speed (float, optional): Maximum value of wind speed for
            contours. Defaults to None.
        cmap (str, optional): Colormap specifier. Defaults to
            'coolwarm'.
        levels (np.array, optional): Contour levels for line contour plot.
            Defaults to None.
        clevels (np.array, optional): Contour levels for tricontourf plot.
            Defaults to None.
        color_bar (Boolean, optional): Flag to include a color bar on the plot.
            Defaults to False.
        title (str, optional): User-supplied title for the plot. Defaults to "".
        **kwargs: Additional parameters to pass to line contour plot.

    Returns:
        im (:py:class:`matplotlib.plt.pcolormesh`): Image handle.
    """

    if not ax:
        fig, ax = plt.subplots()

    if vel_component=='u':
        # vel_mesh = cut_plane.df.u.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
        if min_speed is None:
            min_speed = cut_plane.df.u.min()
        if max_speed is None:
            max_speed = cut_plane.df.u.max()
    elif vel_component=='v':
        # vel_mesh = cut_plane.df.v.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
        if min_speed is None:
            min_speed = cut_plane.df.v.min()
        if max_speed is None:
            max_speed = cut_plane.df.v.max()
    elif vel_component=='w':
        # vel_mesh = cut_plane.df.w.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
        if min_speed is None:
            min_speed = cut_plane.df.w.min()
        if max_speed is None:
            max_speed = cut_plane.df.w.max()

    # Allow separate number of levels for tricontourf and for line_contour
    if clevels is None:
        clevels = levels

    # Plot the cut-through
    im = ax.tricontourf(
        cut_plane.df.x1,
        cut_plane.df.x2,
        cut_plane.df.u,
        vmin=min_speed,
        vmax=max_speed,
        levels=clevels,
        cmap=cmap,
        extend="both",
    )

    if cut_plane.normal_vector == "x":
        ax.invert_xaxis()

    if color_bar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('m/s')

    # Set the title
    ax.set_title(title)

    # Make equal axis
    ax.set_aspect("equal")

    return im



def visualize_turbines(
    farm: Farm,
    ax: plt.Axes = None,
    color: str = None,
    wd: np.ndarray = None,
    yaw_angles: np.ndarray = None,
):
    """
    Plot the wind plant layout from turbine locations gotten from a FlorisInterface object.
    Note that this function automatically uses the first wind direction and first wind speed.
    Generally, it is most explicit to create a new FlorisInterface with only the single
    wind condition that should be plotted.

    Args:
        fi (:py:class:`floris.tools.floris_interface.FlorisInterface`): FlorisInterface object.
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes. Defaults to None.
        color (str, optional): Color to plot turbines. Defaults to None.
        wd (list, optional): The wind direction to plot the turbines relative to. Defaults to None.
        yaw_angles (NDArray, optional): The yaw angles for the turbines. Defaults to None.
    """
    if not ax:
        fig, ax = plt.subplots()
    if yaw_angles is None:
        yaw_angles = farm.yaw_angle
    if wd is None:
        wd = np.array(farm.wind_direction)

    # Rotate yaw angles to inertial frame for plotting turbines relative to wind direction
    yaw_angles = yaw_angles - (wd - 270) % 360

    if color is None:
        color = "k"

    rotor_diameters = farm.turbine.rotor_diameter
    layout = np.array(farm.layout).T
    for x, y, yaw, in zip(layout[0], layout[1], yaw_angles):
        R = rotor_diameters / 2.0
        x_0 = x + np.sin(np.deg2rad(yaw)) * R
        x_1 = x - np.sin(np.deg2rad(yaw)) * R
        y_0 = y - np.cos(np.deg2rad(yaw)) * R
        y_1 = y + np.cos(np.deg2rad(yaw)) * R
        ax.plot([x_0, x_1], [y_0, y_1], color=color)