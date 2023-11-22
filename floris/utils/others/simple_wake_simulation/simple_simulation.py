import os
import copy
import yaml
import numpy as np
# import pandas as pd
import numpy.typing as npt
# import matplotlib.pyplot as plt

from attrs import define, field
from scipy.interpolate import interp1d

NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int_]
NDArrayObject = npt.NDArray[np.object_]


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


def property_assign(dict, param):
    return dict[param]


class VelocityModel():
    def __init__(self) -> None:
        self.we = 0.05

    def prepare(self, grid):
        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
        }
        return kwargs

    def compute(self,
                x_i: np.ndarray,
                y_i: np.ndarray,
                z_i: np.ndarray,
                axial_induction_i: np.ndarray,
                deflection_field_i: np.ndarray,
                yaw_angle_i: np.ndarray,
                turbulence_intensity_i: np.ndarray,
                ct_i: np.ndarray,
                hub_height_i,
                rotor_diameter_i,
                *,
                x: np.ndarray,
                y: np.ndarray,
                z: np.ndarray,
                ) -> None:

        rotor_radius = rotor_diameter_i / 2.0

        # Numexpr - do not change below without corresponding changes above.
        dx = x - x_i
        dy = y - y_i - deflection_field_i
        dz = z - z_i

        we = self.we
        NUM_EPS = 0.001

        # Construct a boolean mask to include all points downstream of the turbine
        downstream_mask = dx > 0 + NUM_EPS

        boundary_mask = np.sqrt(dy ** 2 + dz ** 2) < we * dx + rotor_radius

        # Calculate C for points within the mask and fill points outside with 0
        c = np.where(
            np.logical_and(downstream_mask, boundary_mask),
            (rotor_radius / (rotor_radius + we * dx + NUM_EPS)) ** 2,  # This is "C"
            0.0,
        )

        velocity_deficit = 2 * axial_induction_i * c

        return velocity_deficit


class DeflectionModel():
    def __init__(self) -> None:
        self.kd = 0.05
        self.ad = 0.0
        self.bd = 0.0

    def prepare(self, grid):
        kwargs = {
            "x": grid.x_sorted,
        }
        return kwargs

    def compute(self,
                x_i: np.ndarray,
                y_i: np.ndarray,
                yaw_i: np.ndarray,
                turbulence_intensity_i: np.ndarray,
                ct_i: np.ndarray,
                rotor_diameter_i: np.ndarray,
                *,
                x: np.ndarray,) -> None:
        # angle of deflection
        xi_init = cosd(yaw_i) * sind(yaw_i) * ct_i / 2.0
        delta_x = x - x_i

        # yaw displacement
        A = 15 * (2 * self.kd * delta_x / rotor_diameter_i + 1) ** 4.0 + xi_init ** 2.0
        B = (30 * self.kd / rotor_diameter_i)
        B = B * ( 2 * self.kd * delta_x / rotor_diameter_i + 1 ) ** 5.0
        C = xi_init * rotor_diameter_i * (15 + xi_init ** 2.0)
        D = 30 * self.kd

        yYaw_init = (xi_init * A / B) - (C / D)

        # corrected yaw displacement with lateral offset
        # This has the same shape as the grid

        deflection = yYaw_init + self.ad + self.bd * delta_x

        return deflection


class TurbulenceModel():
    def __init__(self) -> None:
        self.initial = 0.1
        self.constant = 0.5
        self.ai = 0.8
        self.downstream = -0.32

    def prepare(self) -> dict:
        pass

    def compute(self,
                ambient_TI: float,
                x: np.ndarray,
                x_i: np.ndarray,
                rotor_diameter: float,
                axial_induction: np.ndarray,
                ) -> None:
        # Replace zeros and negatives with 1 to prevent nans/infs
        delta_x = x - x_i

        # TODO: ensure that these fudge factors are needed for different rotations
        upstream_mask = delta_x <= 0.1
        downstream_mask = delta_x > -0.1

        #Keep downstream components   Set upstream to 1.0
        delta_x = delta_x * downstream_mask + np.ones_like(delta_x) * upstream_mask

        # turbulence intensity calculation based on Crespo et. al.
        ti = (self.constant
              * axial_induction ** self.ai
              * ambient_TI ** self.initial
              * (delta_x / rotor_diameter) ** self.downstream
              )
        # Mask the 1 values from above with zeros
        return ti * downstream_mask


class CombinationModel():
    def __init__(self) -> None:
        pass

    def prepare(self) -> dict:
        pass

    def compute(self,
                wake_field: np.ndarray,
                velocity_field: np.ndarray):
        return np.hypot(wake_field, velocity_field)


@define
class FarmConfig():
    # Define the inflow conditions
    wind_speed: float = field(default=8.0)
    wind_direction: float = field(default=270.0)
    yaw_angles: list = field(default=[10., 15.])
    wind_shear: float = field(default=0.12)
    turbulence_intensity: float = field(default=0.07)
    air_density: float = field(default=1.225)

    # Define the farm layout
    layout: list = field(default=[[0., 0.], [500., 0.]])
    n_turbines: int = field()
    @n_turbines.default
    def layout_turbine_number(self):
        return len(self.layout)

    # Define the turbine configuration
    turbine_config: DotDict = field()
    @turbine_config.default
    def turbine_config_default(self):
        with open('nrel_5MW.yaml') as fid:
            return DotDict(yaml.load(fid, Loader))

    power_thrust_table: dict = field()
    @power_thrust_table.default
    def power_thrust_table_default(self):
        return DotDict(self.turbine_config.power_thrust_table)

    rotor_diameter: float = field(init=False)
    hub_height: float = field(init=False)
    pP: float = field(init=False)
    pT: float = field(init=False)
    generator_efficiency: float = field(init=False)
    ref_density_cp_ct: float = field(init=False)

    rotor_radius: float = field(init=False)
    rotor_area: float = field(init=False)
    fCp_interp: interp1d = field(init=False)
    fCt_interp: interp1d = field(init=False)
    power_interp: interp1d = field(init=False)

    # Wake field grids property
    grid_resolution: int = field(default=3)
    turbine_coordinates_array: np.array = field(init=False)
    reference_turbine_diameter: np.array = field(init=False)
    sorted_indices: NDArrayInt = field(init=False)
    sorted_coord_indices: NDArrayInt = field(init=False)
    unsorted_indices: NDArrayInt = field(init=False)
    x_center_of_rotation: NDArrayFloat = field(init=False)
    y_center_of_rotation: NDArrayFloat = field(init=False)
    x_sorted: NDArrayFloat = field(init=False)
    y_sorted: NDArrayFloat = field(init=False)
    z_sorted: NDArrayFloat = field(init=False)
    x_sorted_inertial_frame: NDArrayFloat = field(init=False)
    y_sorted_inertial_frame: NDArrayFloat = field(init=False)
    z_sorted_inertial_frame: NDArrayFloat = field(init=False)
    average_method = "cubic-mean"
    yaw_angles_sorted: NDArrayFloat = field(init=False)

    u_initial_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    v_initial_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    w_initial_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    u_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    v_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    w_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    u: NDArrayFloat = field(init=False, default=np.array([]))
    v: NDArrayFloat = field(init=False, default=np.array([]))
    w: NDArrayFloat = field(init=False, default=np.array([]))
    turbulence_intensity_field: NDArrayFloat = field(init=False, default=np.array([]))
    turbulence_intensity_field_sorted: NDArrayFloat = field(init=False, default=np.array([]))
    turbulence_intensity_field_sorted_avg: NDArrayFloat = field(init=False, default=np.array([]))

    velocity: VelocityModel = field(init=False)
    deflection: DeflectionModel = field(init=False)
    turbulence: TurbulenceModel = field(init=False)
    combination: CombinationModel = field(init=False)

    def __attrs_post_init__(self) -> None:

        self.rotor_diameter = self.turbine_config.rotor_diameter
        self.hub_height = self.turbine_config.hub_height
        self.pP = self.turbine_config.pP
        self.pT = self.turbine_config.pT
        self.generator_efficiency = self.turbine_config.generator_efficiency
        self.ref_density_cp_ct = self.turbine_config.ref_density_cp_ct

        self.rotor_radius = self.rotor_diameter / 2.0
        self.rotor_area = np.pi * self.rotor_radius ** 2.0

        self.yaw_angles = np.array(self.yaw_angles)
        wind_speeds = np.array(self.power_thrust_table.wind_speed)
        self.fCp_interp = interp1d(
            wind_speeds,
            self.power_thrust_table.power,
            fill_value=(0.0, 1.0),
            bounds_error=False,
        )
        inner_power = (
            0.5 * self.rotor_area
            * self.fCp_interp(wind_speeds)
            * self.generator_efficiency
            * wind_speeds ** 3
        )
        self.power_interp = interp1d(
            wind_speeds,
            inner_power,
            bounds_error=False,
            fill_value=0
        )
        self.fCt_interp = interp1d(
            wind_speeds,
            self.power_thrust_table.thrust,
            fill_value=(0.0001, 0.9999),
            bounds_error=False,
        )

        self.turbine_coordinates_array = np.concatenate(
            [np.array(self.layout).T, np.ones((1, len(self.layout))) * self.hub_height],
            axis=0
            ).T
        self.reference_turbine_diameter = np.ones(self.n_turbines) * self.rotor_diameter

        x, y, z, self.x_center_of_rotation, self.y_center_of_rotation = \
            rotate_coordinates_rel_west(
                self.wind_direction,
                self.turbine_coordinates_array,
        )

        radius_ratio = 0.5
        disc_area_radius = radius_ratio * self.reference_turbine_diameter / 2
        template_grid = np.ones(
            (
                self.n_turbines,
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
            (self.n_turbines, self.grid_resolution, self.grid_resolution),
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

        self.yaw_angles_sorted = np.take_along_axis(
            self.yaw_angles, self.sorted_coord_indices, axis=0,
            )

        self.velocity = VelocityModel()
        self.deflection = DeflectionModel()
        self.turbulence = TurbulenceModel()
        self.combination = CombinationModel()

        wind_profile_plane = (self.z_sorted / self.hub_height) ** self.wind_shear
        dwind_profile_plane = (
            self.wind_shear
            * (1 / self.hub_height) ** self.wind_shear
            * (self.z_sorted) ** (self.wind_shear - 1)
        )

        self.wind_speed = np.array([self.wind_speed])
        self.u_initial_sorted = (
            (self.wind_speed[None, :].T * wind_profile_plane.T).T
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

        self.turbulence_intensity_field = \
            self.turbulence_intensity * np.ones((self.n_turbines, 1, 1))
        self.turbulence_intensity_field_sorted = \
            self.turbulence_intensity_field.copy()

    def turbine_power(self, ):
        deflection_args = self.deflection.prepare(self)
        deficit_args = self.velocity.prepare(self)

        wake_field = np.zeros_like(self.u_initial_sorted)
        v_wake = np.zeros_like(self.v_initial_sorted)
        w_wake = np.zeros_like(self.w_initial_sorted)

        turbine_turbulence_intensity = (
            self.turbulence_intensity
            * np.ones((self.n_turbines, 1, 1))
        )
        ambient_turbulence_intensity = self.turbulence_intensity

        for i in range(self.n_turbines):
            # Get the current turbine quantities
            x_i = np.mean(self.x_sorted[i:i+1], axis=(1, 2))
            x_i = x_i[:, None, None]
            y_i = np.mean(self.y_sorted[i:i+1], axis=(1, 2))
            y_i = y_i[:, None, None]
            z_i = np.mean(self.z_sorted[i:i+1], axis=(1, 2))
            z_i = z_i[:, None, None]

            u_i = self.u_sorted[i:i+1]
            v_i = self.v_sorted[i:i+1]
            yaw_angle_i = self.yaw_angles_sorted[i:i+1, None, None]

            avg_vel = np.cbrt(np.mean(self.u_sorted[i:i+1] ** 3.0, axis=(1, 2)))
            ct_i = np.clip(self.fCt_interp(avg_vel), 0.0001, 0.9999)
            ct_i = ct_i * cosd(yaw_angle_i)
            axial_ind_i = 0.5 / cosd(yaw_angle_i) * (1 - np.sqrt(1 - ct_i * cosd(yaw_angle_i)))
            turb_intensity_i = turbine_turbulence_intensity[i:i+1]
            rotor_diameter_i = np.ones_like(ct_i) * self.rotor_diameter
            hub_height_i = np.ones_like(ct_i) * self.hub_height

            effective_yaw_i = np.zeros_like(yaw_angle_i)
            effective_yaw_i += yaw_angle_i

            deflection_field = self.deflection.compute(
                x_i,
                y_i,
                effective_yaw_i,
                turb_intensity_i,
                ct_i,
                rotor_diameter_i,
                **deflection_args,
            )

            velocity_deficit = self.velocity.compute(
                x_i,
                y_i,
                z_i,
                axial_ind_i,
                deflection_field,
                yaw_angle_i,
                turb_intensity_i,
                ct_i,
                hub_height_i,
                rotor_diameter_i,
                **deficit_args,
            )

            wake_field = self.combination.compute(
                wake_field,
                velocity_deficit * self.u_initial_sorted
            )

            wake_added_turbulence_intensity = self.turbulence.compute(
                ambient_turbulence_intensity,
                self.x_sorted,
                x_i,
                rotor_diameter_i,
                axial_ind_i,
            )

            # Calculate wake overlap for wake-added turbulence (WAT)
            area_overlap = (
                np.sum(velocity_deficit * self.u_initial_sorted > 0.05, axis=(1, 2))
                / (self.grid_resolution * self.grid_resolution)
                )[:, None, None]

            # Modify wake added turbulence by wake area overlap
            downstream_influence_length = 15 * rotor_diameter_i
            ti_added = (
                area_overlap
                * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
                * (self.x_sorted > x_i)
                * (np.abs(y_i - self.y_sorted) < 2 * rotor_diameter_i)
                * (self.x_sorted <= downstream_influence_length + x_i)
                )

            # Combine turbine TIs with WAT
            turbine_turbulence_intensity = np.maximum(
                np.sqrt( ti_added ** 2 + ambient_turbulence_intensity ** 2 ),
                turbine_turbulence_intensity
                )

            self.u_sorted = self.u_initial_sorted - wake_field
            self.v_sorted += v_wake
            self.w_sorted += w_wake

        self.turbulence_intensity_field_sorted = turbine_turbulence_intensity
        self.turbulence_intensity_field_sorted_avg = np.mean(
            turbine_turbulence_intensity, axis=(1, 2))[:, None, None]

        self.u = np.take_along_axis(self.u_sorted, self.unsorted_indices, axis=0)
        self.v = np.take_along_axis(self.v_sorted, self.unsorted_indices, axis=0)
        self.w = np.take_along_axis(self.w_sorted, self.unsorted_indices, axis=0)

        self.turbulence_intensity_field = np.mean(
            np.take_along_axis(
                self.turbulence_intensity_field_sorted,
                self.unsorted_indices,
                axis=0
            ),
            axis=(1, 2)
        )

        return self.compute_power()

    def compute_power(self, ):
        avg_vel = np.cbrt(np.mean(self.u ** 3.0, axis=(1, 2)))
        effective_vel = (self.air_density / self.ref_density_cp_ct) ** (1/3) * avg_vel
        # Compute the rotor effective velocity adjusting for yaw settings
        p = self.power_interp(effective_vel * cosd(self.yaw_angles) ** (self.pP / 3.0))
        return p * self.ref_density_cp_ct


def rotate_coordinates_rel_west(
    wind_directions,
    coordinates,
    x_center_of_rotation=None,
    y_center_of_rotation=None
):
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
    return np.cos(np.radians(angle))


def sind(angle):
    return np.sin(np.radians(angle))


def tand(angle):
    return np.tan(np.radians(angle))



if __name__ == '__main__':
    farm = FarmConfig()
    turbine_power = farm.turbine_power()
    print(turbine_power / 1000.)