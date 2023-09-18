import os
import yaml
import copy
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from tool import (
    State,
    turbine_coordinates,
    rotate_coordinates_rel_west,
    reverse_rotate_coordinates_rel_west)


class Loader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super().__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, self.__class__)

Loader.add_constructor('!include', Loader.include)

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                value = DotDict(value)
            return value
        except:
            return super().__getattribute__(key)

    def __deepcopy__(self, memo=None, _nil=[]):
        if memo is None:
            memo = {}
        d = id(self)
        y = memo.get(d, _nil)
        if y is not _nil:
            return y

        dict = DotDict()
        memo[d] = id(dict)
        for key in self.keys():
            dict.__setattr__(copy.deepcopy(key, memo),
                             copy.deepcopy(self.__getattr__(key), memo))
        return dict


class Turbine(DotDict):
    """Default turbine type is NREL 5WM"""
    def __init__(self, config='./nrel_5MW.yaml') -> None:
        super(Turbine, self).__init__()
        self.update(self._load_config(config))
        # self.hub_height = 90.
        # self.rotor_diameter = 126.
        self.setdefault('rotor_area', 0.25 * np.pi * self.rotor_diameter ** 2.0)
        self._construct_thrust_power()

    def _load_config(self, path, loader=Loader) -> None:
        with open(path) as fid:
            return DotDict(yaml.load(fid, loader))

    def _construct_thrust_power(self, ) -> None:
        wind_speeds = np.array(self.power_thrust_table.wind_speed)
        self.setdefault('fCp_interp', interp1d(
            wind_speeds,
            self.power_thrust_table.power,
            fill_value=(0.0, 1.0),
            bounds_error=False,
            ))
        inner_power = (
            0.5 * self.rotor_area
            * self.fCp_interp(wind_speeds)
            * self.generator_efficiency
            * wind_speeds ** 3
            )
        self.setdefault('power_interp', interp1d(
            wind_speeds,
            inner_power,
            bounds_error=False,
            fill_value=0
            ))
        self.setdefault('fCt_interp', interp1d(
            wind_speeds,
            self.power_thrust_table.thrust,
            fill_value=(0.0001, 0.9999),
            bounds_error=False,
            ))


class Farm():
    state = State.UNINITIALIZED
    def __init__(self, direction, layout, yaw_angle, turbine) -> None:
        self.air_density = 1.225
        self.wind_shear = 0.12
        self.wind_direction = direction
        self.layout = layout
        self.n_turbine = len(layout)
        self.yaw_angle = np.array(yaw_angle)
        self.turbine_list = np.array([turbine() for _ in range(self.n_turbine)])
        self.turbine = self.turbine_list[0]

    def turbine_property_sort(self, sorted_coord_indices) -> None:
        self.turbine_list_sorted = np.take_along_axis(
            self.turbine_list, sorted_coord_indices, axis=0,
            )

    def yaw_angle_sort(self, sorted_coord_indices) -> None:
        # Sort yaw angles from most upstream to most downstream wind turbine
        self.yaw_angle_sorted = np.take_along_axis(
            self.yaw_angle, sorted_coord_indices, axis=0,
            )
        self.state = State.INITIALIZED

    def finalize(self, unsorted_indices) -> None:
        self.state.USED
        pass


class TurbineGrid():
    """Grid class for storing turbine rotor points to calculate power"""
    def __init__(self, direction, layout, farm, grid_resolution=3) -> None:
        self.n_turbine = len(layout)
        self.turbine = farm.turbine
        self.wind_direction = direction
        self.grid_resolution = grid_resolution
        self.turbine_coordinates_array = \
            turbine_coordinates(layout, self.turbine.hub_height)
        self.reference_turbine_diameter = \
            np.ones(len(layout)) * self.turbine.rotor_diameter

        self.set_grid()

    def set_grid(self, ) -> None:

        x, y, z, self.x_center_of_rotation, self.y_center_of_rotation = \
            rotate_coordinates_rel_west(
                self.wind_direction,
                self.turbine_coordinates_array,
        )

        radius_ratio = 0.5
        disc_area_radius = radius_ratio * self.reference_turbine_diameter / 2
        template_grid = np.ones(
            (
                self.n_turbine,
                self.grid_resolution,
                self.grid_resolution,
            ),
            dtype=np.float64
        )

        disc_grid = np.linspace(
                -1 * disc_area_radius,
                disc_area_radius,
                self.grid_resolution,
                dtype=np.float64,
                axis=1
            )

        ones_grid = np.ones(
            (self.n_turbine, self.grid_resolution, self.grid_resolution),
            dtype=np.float64
        )
        _x = x[:, None, None] * template_grid
        _y = y[:, None, None] + template_grid * (disc_grid[:, :, None])
        _z = z[:, None, None] + template_grid * (disc_grid[:, None, :] * ones_grid)

        self.sorted_indices = _x.argsort(axis=0)
        self.sorted_coord_indices = x.argsort(axis=0)
        self.unsorted_indices = self.sorted_indices.argsort(axis=0)

        self.x_sorted = np.take_along_axis(_x, self.sorted_indices, axis=0)
        self.y_sorted = np.take_along_axis(_y, self.sorted_indices, axis=0)
        self.z_sorted = np.take_along_axis(_z, self.sorted_indices, axis=0)

        self.x_sorted_inertial_frame, self.y_sorted_inertial_frame, self.z_sorted_inertial_frame = \
            reverse_rotate_coordinates_rel_west(
                wind_directions=self.wind_direction,
                grid_x=self.x_sorted,
                grid_y=self.y_sorted,
                grid_z=self.z_sorted,
                x_center_of_rotation=self.x_center_of_rotation,
                y_center_of_rotation=self.y_center_of_rotation,
            )


class PlanarGrid():
    """Grid class for storing flow field points to calculate and plot wake"""
    def __init__(self, direction, layout, farm, grid_resolution=[200, 100],
                 normal_vector='z', planar_coordinate=None, bounds=None) -> None:
        self.wind_direction = direction
        self.n_turbine = len(layout)
        self.turbine = farm.turbine
        self.grid_resolution = grid_resolution
        self.normal_vector = normal_vector
        self.planar_coordinate = planar_coordinate
        self.bounds = bounds
        self.turbine_coordinates_array = \
            turbine_coordinates(layout, self.turbine.hub_height)
        self.reference_turbine_diameter = \
            np.ones(len(layout)) * self.turbine.rotor_diameter

        self.set_grid()

    def set_grid(self, ) -> None:
        x, y, z, self.x_center_of_rotation, self.y_center_of_rotation = rotate_coordinates_rel_west(
            self.wind_direction,
            self.turbine_coordinates_array
        )
        max_diameter = np.max(self.reference_turbine_diameter)

        if self.normal_vector == "z":  # Rules of thumb for horizontal plane
            # Using original x and y coordinates to determine the bounds of the grid
            x, y, z = self.turbine_coordinates_array.T
            if self.bounds is None:
                self.x1_bounds = (np.min(x) - 2 * max_diameter, np.max(x) + 10 * max_diameter)
                self.x2_bounds = (np.min(y) - 2 * max_diameter, np.max(y) + 2 * max_diameter)
            else:
                self.x1_bounds, self.x2_bounds = self.bounds

            # TODO figure out proper z spacing for GCH, currently set to +/- 10.0
            x_points, y_points, z_points = np.meshgrid(
                np.linspace(self.x1_bounds[0], self.x1_bounds[1], int(self.grid_resolution[0])),
                np.linspace(self.x2_bounds[0], self.x2_bounds[1], int(self.grid_resolution[1])),
                np.array([
                    float(self.planar_coordinate) - 10.0,
                    float(self.planar_coordinate),
                    float(self.planar_coordinate) + 10.0
                ]),
                indexing="ij"
            )

            # Rotate the grid coordinates to the reference orientation along the opposite direction
            self.x_sorted, self.y_sorted, self.z_sorted = \
            reverse_rotate_coordinates_rel_west(
                wind_directions=self.wind_direction,
                grid_x=x_points[None, None, :, :, :],
                grid_y=y_points[None, None, :, :, :],
                grid_z=z_points[None, None, :, :, :],
                x_center_of_rotation=self.x_center_of_rotation,
                y_center_of_rotation=self.y_center_of_rotation,
                reverse_rotate_dir=1.0
            )

            # self.x_sorted = x_points[None, None, :, :, :]
            # self.y_sorted = y_points[None, None, :, :, :]
            # self.z_sorted = z_points[None, None, :, :, :]

        elif self.normal_vector == "x":  # Rules of thumb for cross plane
            if self.bounds is None:
                self.x1_bounds = (np.min(y) - 2 * max_diameter, np.max(y) + 2 * max_diameter)
                self.x2_bounds = (0.001, 6 * np.max(z))
            else:
                self.x1_bounds, self.x2_bounds = self.bounds

            x_points, y_points, z_points = np.meshgrid(
                np.array([float(self.planar_coordinate)]),
                np.linspace(self.x1_bounds[0], self.x1_bounds[1], int(self.grid_resolution[0])),
                np.linspace(self.x2_bounds[0], self.x2_bounds[1], int(self.grid_resolution[1])),
                indexing="ij"
            )

            self.x_sorted = x_points[None, None, :, :, :]
            self.y_sorted = y_points[None, None, :, :, :]
            self.z_sorted = z_points[None, None, :, :, :]

        elif self.normal_vector == "y":  # Rules of thumb for y plane
            if self.x1_bounds is None:
                self.x1_bounds = (np.min(x) - 2 * max_diameter, np.max(x) + 10 * max_diameter)
                self.x2_bounds = (0.001, 6 * np.max(z))
            else:
                self.x1_bounds, self.x2_bounds = self.bounds

            x_points, y_points, z_points = np.meshgrid(
                np.linspace(self.x1_bounds[0], self.x1_bounds[1], int(self.grid_resolution[0])),
                np.array([float(self.planar_coordinate)]),
                np.linspace(self.x2_bounds[0], self.x2_bounds[1], int(self.grid_resolution[1])),
                indexing="ij"
            )

            self.x_sorted = x_points[None, None, :, :, :]
            self.y_sorted = y_points[None, None, :, :, :]
            self.z_sorted = z_points[None, None, :, :, :]

        # Now calculate grid coordinates in original frame (from 270 deg perspective)
        self.x_sorted_inertial_frame, self.y_sorted_inertial_frame, self.z_sorted_inertial_frame = \
            reverse_rotate_coordinates_rel_west(
                wind_directions=self.wind_direction,
                grid_x=self.x_sorted,
                grid_y=self.y_sorted,
                grid_z=self.z_sorted,
                x_center_of_rotation=self.x_center_of_rotation,
                y_center_of_rotation=self.y_center_of_rotation,
            )


class PointsGrid():
    def __init__(self) -> None:
        pass

    def set_grid(self, ) -> None:
        pass


class FlowField():
    """Flow data storage"""
    def __init__(self, velocity, direction, turbulence, layout, grid, farm) -> None:
        self.wind_speed = np.array([velocity])
        self.wind_direction = direction
        self.wind_turbulence = turbulence
        self.turbine = farm.turbine
        self.n_turbine = len(layout)
        self.wind_shear = farm.wind_shear

        self.init_field(grid)

    def init_field(self, grid):
        self.vel_field_init(grid)
        self.turb_field_init(grid)

    def vel_field_init(self, grid):
        wind_profile_plane = (grid.z_sorted / self.turbine.hub_height) ** self.wind_shear
        dwind_profile_plane = (
            self.wind_shear
            * (1 / self.turbine.hub_height) ** self.wind_shear
            * (grid.z_sorted) ** (self.wind_shear - 1)
        )

        self.u_initial_sorted = (
            (self.wind_speed[None, :].T * wind_profile_plane.T).T
            )
        self.dudz_initial_sorted = (
            (self.wind_speed[None, :].T * dwind_profile_plane.T).T
            )
        self.v_initial_sorted = np.zeros(
            np.shape(self.u_initial_sorted),
            dtype=self.u_initial_sorted.dtype
            )
        self.w_initial_sorted = np.zeros(
            np.shape(self.u_initial_sorted),
            dtype=self.u_initial_sorted.dtype
            )

        self.u_sorted = self.u_initial_sorted.copy()
        self.v_sorted = self.v_initial_sorted.copy()
        self.w_sorted = self.w_initial_sorted.copy()

    def turb_field_init(self, grid):
        self.turbulence_intensity_field = \
            self.wind_turbulence * np.ones((self.n_turbine, 1, 1))
        self.turbulence_intensity_field_sorted = \
            self.turbulence_intensity_field.copy()

    def finalize(self, unsorted_indices) -> None:
        self.u = np.take_along_axis(self.u_sorted, unsorted_indices, axis=0)
        self.v = np.take_along_axis(self.v_sorted, unsorted_indices, axis=0)
        self.w = np.take_along_axis(self.w_sorted, unsorted_indices, axis=0)

        self.turbulence_intensity_field = np.mean(
            np.take_along_axis(
                self.turbulence_intensity_field_sorted,
                unsorted_indices,
                axis=0
            ),
            axis=(1, 2)
        )


class CutPlane:
    """
    A CutPlane object represents a 2D slice through the flow of a
    FLORIS simulation, or other such as SOWFA result.
    """

    def __init__(self, df, x1_resolution, x2_resolution, normal_vector):
        """
        Initialize CutPlane object, storing the DataFrame and resolution.

        Args:
            df (pandas.DataFrame): Pandas DataFrame of data with
                columns x1, x2, u, v, w.
        """
        self.df: pd.DataFrame = df
        self.normal_vector: str = normal_vector
        self.resolution = (x1_resolution, x2_resolution)
        self.df.set_index(["x1", "x2"])

    def __sub__(self, other):

        if self.normal_vector != other.normal_vector:
            raise ValueError("Operands must have consistent normal vectors.")

        # if self.normal_vector.df.
        # DF must be of the same size
        # resolution must be of the same size

        df: pd.DataFrame = self.df.copy()
        other_df: pd.DataFrame = other.df.copy()

        df['u'] = self.df['u'] - other_df['u']
        df['v'] = self.df['v'] - other_df['v']
        df['w'] = self.df['w'] - other_df['w']

        return CutPlane(
            df,
            self.resolution[0],
            self.resolution[1],
            self.normal_vector
        )