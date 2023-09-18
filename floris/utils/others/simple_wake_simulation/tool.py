import logging
import numpy as np
import numpy.typing as npt

from enum import Enum


NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int_]
NDArrayObject = npt.NDArray[np.object_]


class State(Enum):
    UNINITIALIZED = 0
    INITIALIZED = 1
    USED = 2


class LoggerBase:
    """
    Convenience super-class to any class requiring access to the logging
    module. The virtual property here allows a simple and dynamic method
    for obtaining the correct logger for the calling class.
    """

    @property
    def logger(self):
        return logging.getLogger(
            "{}.{}".format(type(self).__module__, type(self).__name__)
        )


def turbine_coordinates(layout, height):
    return np.concatenate(
        [np.array(layout).T, np.ones((1, len(layout))) * height],
        axis=0).T


def rotate_coordinates_rel_west(
    wind_directions,
    coordinates,
    x_center_of_rotation=None,
    y_center_of_rotation=None
):
    """
    This function rotates the given coordinates so that they are aligned with West (270) rather
    than North (0). The rotation happens about the centroid of the coordinates.

    Args:
        wind_directions (NDArrayFloat): Series of wind directions to base the rotation.
        coordinates (NDArrayFloat): Series of coordinates to rotate with shape (N coordinates, 3)
            so that each element of the array coordinates[i] yields a three-component coordinate.
        x_center_of_rotation (float, optional): The x-coordinate for the rotation center of the
            input coordinates. Defaults to None.
        y_center_of_rotation (float, optional): The y-coordinate for the rotational center of the
            input coordinates. Defaults to None.
    """

    # Calculate the difference in given wind direction from 270 / West
    wind_deviation_from_west = (wind_directions - 270) % 360
    # wind_deviation_from_west = np.reshape(wind_deviation_from_west, (1, 1, 1))

    # Construct the arrays storing the turbine locations
    x_coordinates, y_coordinates, z_coordinates = coordinates.T
    # print(coordinates)

    # Find center of rotation - this is the center of box bounding all of the turbines
    if x_center_of_rotation is None:
        x_center_of_rotation = (np.min(x_coordinates) + np.max(x_coordinates)) / 2
    if y_center_of_rotation is None:
        y_center_of_rotation = (np.min(y_coordinates) + np.max(y_coordinates)) / 2

    # Rotate turbine coordinates about the center
    x_coord_offset = x_coordinates - x_center_of_rotation
    y_coord_offset = y_coordinates - y_center_of_rotation
    x_coord_rotated = (
        x_coord_offset * cosd(wind_deviation_from_west)
        - y_coord_offset * sind(wind_deviation_from_west)
        + x_center_of_rotation
    )
    y_coord_rotated = (
        x_coord_offset * sind(wind_deviation_from_west)
        + y_coord_offset * cosd(wind_deviation_from_west)
        + y_center_of_rotation
    )
    z_coord_rotated = np.ones_like(wind_deviation_from_west) * z_coordinates
    return x_coord_rotated, y_coord_rotated, z_coord_rotated, x_center_of_rotation, \
        y_center_of_rotation


def reverse_rotate_coordinates_rel_west(
    wind_directions: NDArrayFloat,
    grid_x: NDArrayFloat,
    grid_y: NDArrayFloat,
    grid_z: NDArrayFloat,
    x_center_of_rotation: float,
    y_center_of_rotation: float,
    reverse_rotate_dir: float = -1.0
):
    """
    This function reverses the rotation of the given grid so that the coordinates are aligned with
    the original wind direction. The rotation happens about the centroid of the coordinates.

    Args:
        wind_directions (NDArrayFloat): Series of wind directions to base the rotation.
        coordinates (NDArrayFloat): Series of coordinates to rotate with shape (N coordinates, 3)
            so that each element of the array coordinates[i] yields a three-component coordinate.
        grid_x (NDArrayFloat): X-coordinates to be rotated.
        grid_y (NDArrayFloat): Y-coordinates to be rotated.
        grid_z (NDArrayFloat): Z-coordinates to be rotated.
        x_center_of_rotation (float): The x-coordinate for the rotation center of the
            input coordinates.
        y_center_of_rotation (float): The y-coordinate for the rotational center of the
            input coordinates.
    """
    # Calculate the difference in given wind direction from 270 / West
    # We are rotating in the other direction
    wind_deviation_from_west = reverse_rotate_dir * ((wind_directions - 270) % 360)

    # Construct the arrays storing the turbine locations
    grid_x_reversed = np.zeros_like(grid_x)
    grid_y_reversed = np.zeros_like(grid_x)
    grid_z_reversed = np.zeros_like(grid_x)

    x_rot = grid_x[:, :, :]
    y_rot = grid_y[:, :, :]
    z_rot = grid_z[:, :, :]

    # Rotate turbine coordinates about the center
    x_rot_offset = x_rot - x_center_of_rotation
    y_rot_offset = y_rot - y_center_of_rotation
    x = (
        x_rot_offset * cosd(wind_deviation_from_west)
        - y_rot_offset * sind(wind_deviation_from_west)
        + x_center_of_rotation
    )
    y = (
        x_rot_offset * sind(wind_deviation_from_west)
        + y_rot_offset * cosd(wind_deviation_from_west)
        + y_center_of_rotation
    )
    z = z_rot  # Nothing changed in this rotation

    grid_x_reversed[:, :, :] = x
    grid_y_reversed[:, :, :] = y
    grid_z_reversed[:, :, :] = z

    # print(grid_x_reversed)
    return grid_x_reversed, grid_y_reversed, grid_z_reversed


def cosd(angle):
    """
    Cosine of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.cos(np.radians(angle))


def sind(angle):
    """
    Sine of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.sin(np.radians(angle))


def tand(angle):
    """
    Tangent of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.tan(np.radians(angle))