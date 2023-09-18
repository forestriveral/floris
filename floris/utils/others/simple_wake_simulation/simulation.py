import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Iterable

from tool import (
    LoggerBase,
    State,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
    cosd,
    sind,
)

from module import (
    Turbine,
    Farm,
    TurbineGrid,
    PlanarGrid,
    PointsGrid,
    FlowField,
    CutPlane,
)

from model import (
    WakeModel,
    VelocityModel,
    DeflectionModel,
    TurbulenceModel,
    CombinationModel,
)

from visualize import (
    visualize_cut_plane,
    visualize_turbines,
)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class WakeSimulator(LoggerBase):
    state = State.UNINITIALIZED
    def __init__(self,
                 velocity=8.,
                 direction=270.,
                 turbulence=0.07,
                 layout=[[0., 0.], [500., 0.]],
                 yaw_angle=[10., 15.],
                 ) -> None:
        self.wind_vel = velocity
        self.wind_dir = direction
        self.wind_turb = turbulence
        self.layout = layout
        self.yaw_angle = yaw_angle
        self.turbine = Turbine
        self.farm = self.farm_map()
        self.model = self.model_manager()
        self.grid = None
        self.flow = None

    def farm_map(self, ) -> None:
        return Farm(
            direction=self.wind_dir,
            layout=self.layout,
            yaw_angle=self.yaw_angle,
            turbine=self.turbine,
        )

    def turbine_grid(self, ) -> None:
        self.grid = TurbineGrid(
            direction=self.wind_dir,
            layout=self.layout,
            farm=self.farm,
            grid_resolution=3,
            )
        self.farm.turbine_property_sort(
            self.grid.sorted_coord_indices)

    def flow_grid(self, settings) -> None:
        if settings['type'] == 'planar':
            self.grid = PlanarGrid(
                direction=self.wind_dir,
                layout=self.layout,
                farm=self.farm,
                normal_vector='z',
                grid_resolution=settings['resolution'],
                planar_coordinate=settings['coordinate'],
                bounds=settings['bounds'],
            )
        else:
            self.grid = PointsGrid(
                direction=self.wind_dir,
                layout=self.layout,
                farm=self.farm,
            )

    def flow_field(self, ):
        if self.grid is None:
            raise ValueError('Turbine grid not initialized')

        self.flow = FlowField(
            velocity=self.wind_vel,
            direction=self.wind_dir,
            turbulence=self.wind_turb,
            layout=self.layout,
            grid=self.grid,
            farm=self.farm,
            )

        if isinstance(self.grid, TurbineGrid):
            self.farm.yaw_angle_sort(
                self.grid.sorted_coord_indices)

    def model_manager(self, ) -> None:
        return WakeModel(
            velocity_model=VelocityModel(),
            deflection_model=DeflectionModel(),
            turbulence_model=TurbulenceModel(),
            combination_model=CombinationModel(),
        )

    def wake_solver(self, type='turbine') -> None:
        default_solver = {
            'turbine': turbine_solver,
            'field': field_solver,
        }
        return default_solver[type](
            farm=self.farm,
            flow=self.flow,
            grid=self.grid,
            model=self.model,
            )

    def power_calculation(self, ) -> NDArrayFloat:
        # Check for negative velocities, which could indicate bad model
        # parameters or turbines very closely spaced.
        turbine_effective_velocities = rotor_effective_velocity(
            air_density=self.farm.air_density,
            ref_density_cp_ct=self.farm.turbine.ref_density_cp_ct,
            velocities=self.flow.u,
            yaw_angle=self.farm.yaw_angle,
            pP=self.farm.turbine.pP,
            pT=self.farm.turbine.pT,
            turbine_type_map=self.farm.turbine_list,
        )
        if (turbine_effective_velocities < 0.).any():
            self.logger.warning("Some rotor effective velocities are negative.")

        turbine_powers = turbine_power_calculation(
            ref_density_cp_ct=self.farm.turbine.ref_density_cp_ct,
            rotor_effective_velocities=turbine_effective_velocities,
            power_interp=self.farm.turbine.power_interp,
            turbine_type_map=self.farm.turbine_list,
        )
        return turbine_powers

    def finalize(self, ) -> None:
        self.flow.finalize(self.grid.unsorted_indices)
        self.farm.finalize(self.grid.unsorted_indices)
        self.state = State.USED

    @property
    def turbine_power(self,) -> None:
        self.turbine_grid()
        self.flow_field()
        self.state.INITIALIZED

        self.wake_solver('turbine')
        self.finalize()
        return self.power_calculation()

    def wake_plot(self, plane) -> None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

        visualize_cut_plane(
            plane,
            ax=ax,
            vel_component='u',
            min_speed=None,
            max_speed=None,
            cmap="coolwarm",
            levels=None,
            clevels=None,
            color_bar=False,
            title="",
        )

        visualize_turbines(self.farm, ax=ax)

        plt.show()

    def wake_field(self, settings=None, plot=True) -> None:
        default_settings = {
            "type": "planar",
            "normal_vector": "z",
            "coordinate": self.farm.turbine.hub_height,
            "resolution": [200, 100],
            "bounds": None,
        }

        if settings is not None:
            settings = default_settings.update(settings)
        else:
            settings = default_settings

        self.flow_grid(settings)
        self.flow_field()
        self.wake_solver('field')

        df = get_plane_points(
            self.grid,
            self.flow,
            settings['normal_vector'],
            settings['coordinate']
            )

        # Compute the cutplane
        horizontal_plane = CutPlane(
            df,
            settings['resolution'][0],
            settings['resolution'][1],
            settings['normal_vector'],
        )

        if plot:
            self.wake_plot(horizontal_plane)

        return horizontal_plane


def turbine_solver(farm, flow, grid, model):

    deflection_args = model.deflection.prepare(grid, flow)
    deficit_args = model.velocity.prepare(grid, flow)

    wake_field = np.zeros_like(flow.u_initial_sorted)
    v_wake = np.zeros_like(flow.v_initial_sorted)
    w_wake = np.zeros_like(flow.w_initial_sorted)

    turbine_turbulence_intensity = (
        flow.wind_turbulence
        * np.ones((farm.n_turbine, 1, 1))
    )
    ambient_turbulence_intensity = flow.wind_turbulence

    for i in range(grid.n_turbine):
        # Get the current turbine quantities
        x_i = np.mean(grid.x_sorted[i:i+1], axis=(1, 2))
        x_i = x_i[:, None, None]
        y_i = np.mean(grid.y_sorted[i:i+1], axis=(1, 2))
        y_i = y_i[:, None, None]
        z_i = np.mean(grid.z_sorted[i:i+1], axis=(1, 2))
        z_i = z_i[:, None, None]

        u_i = flow.u_sorted[i:i+1]
        v_i = flow.v_sorted[i:i+1]
        yaw_angle_i = farm.yaw_angle_sorted[i:i+1, None, None]
        turbine_i = farm.turbine_list_sorted[i]

        avg_vel = average_velocity(flow.u_sorted[i:i+1])
        ct_i = np.clip(farm.turbine_list_sorted[i].fCt_interp(avg_vel), 0.0001, 0.9999)
        ct_i = ct_i * cosd(yaw_angle_i)
        axial_ind_i = 0.5 / cosd(yaw_angle_i) * (1 - np.sqrt(1 - ct_i * cosd(yaw_angle_i)))
        turb_intensity_i = turbine_turbulence_intensity[i:i+1]
        rotor_diameter_i = np.ones_like(ct_i) * turbine_i.rotor_diameter
        hub_height_i = np.ones_like(ct_i) * turbine_i.hub_height

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        deflection_field = model.deflection.compute(
            x_i,
            y_i,
            effective_yaw_i,
            turb_intensity_i,
            ct_i,
            rotor_diameter_i,
            **deflection_args,
        )

        velocity_deficit = model.velocity.compute(
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

        wake_field = model.combination.compute(
            wake_field,
            velocity_deficit * flow.u_initial_sorted
        )

        wake_added_turbulence_intensity = model.turbulence.compute(
            ambient_turbulence_intensity,
            grid.x_sorted,
            x_i,
            rotor_diameter_i,
            axial_ind_i,
        )

        # Calculate wake overlap for wake-added turbulence (WAT)
        area_overlap = (
            np.sum(velocity_deficit * flow.u_initial_sorted > 0.05, axis=(1, 2))
            / (grid.grid_resolution * grid.grid_resolution)
            )[:, None, None]

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * rotor_diameter_i
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
            * (grid.x_sorted > x_i)
            * (np.abs(y_i - grid.y_sorted) < 2 * rotor_diameter_i)
            * (grid.x_sorted <= downstream_influence_length + x_i)
            )

        # Combine turbine TIs with WAT
        turbine_turbulence_intensity = np.maximum(
            np.sqrt( ti_added ** 2 + ambient_turbulence_intensity ** 2 ),
            turbine_turbulence_intensity
            )

        flow.u_sorted = flow.u_initial_sorted - wake_field
        flow.v_sorted += v_wake
        flow.w_sorted += w_wake

    flow.turbulence_intensity_field_sorted = turbine_turbulence_intensity
    flow.turbulence_intensity_field_sorted_avg = np.mean(
        turbine_turbulence_intensity, axis=(1, 2))[:, None, None]


def field_solver(farm, flow, grid, model):
    # Get the flow quantities and turbine performance
    turbine_farm = copy.deepcopy(farm)
    turbine_flow = copy.deepcopy(flow)
    turbine_grid = TurbineGrid(
        direction=grid.wind_direction,
        layout=turbine_farm.layout,
        farm=turbine_farm,
        grid_resolution=3,
    )
    turbine_farm.turbine_property_sort(
        turbine_grid.sorted_coord_indices)
    turbine_flow.init_field(turbine_grid)
    turbine_farm.yaw_angle_sort(
        turbine_grid.sorted_coord_indices)
    turbine_solver(turbine_farm, turbine_flow, turbine_grid, model)

    deflection_args = model.deflection.prepare(grid, flow)
    deficit_args = model.velocity.prepare(grid, flow)

    wake_field = np.zeros_like(flow.u_initial_sorted)
    v_wake = np.zeros_like(flow.v_initial_sorted)
    w_wake = np.zeros_like(flow.w_initial_sorted)

    # turbine_turbulence_intensity = (
    #     flow.wind_turbulence
    #     * np.ones((farm.n_turbine, 1, 1))
    # )
    # ambient_turbulence_intensity = flow.wind_turbulence

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbine):
        # Get the current turbine quantities
        x_i = np.mean(turbine_grid.x_sorted[i:i+1], axis=(1, 2))
        x_i = x_i[:, None, None]
        y_i = np.mean(turbine_grid.y_sorted[i:i+1], axis=(1, 2))
        y_i = y_i[:, None, None]
        z_i = np.mean(turbine_grid.z_sorted[i:i+1], axis=(1, 2))
        z_i = z_i[:, None, None]

        u_i = turbine_flow.u_sorted[i:i+1]
        v_i = turbine_flow.v_sorted[i:i+1]
        yaw_angle_i = turbine_farm.yaw_angle_sorted[i:i+1, None, None]
        turbine_i = turbine_farm.turbine_list_sorted[i]

        avg_vel = average_velocity(turbine_flow.u_sorted[i:i+1])
        ct_i = np.clip(turbine_farm.turbine_list_sorted[i].fCt_interp(avg_vel), 0.0001, 0.9999)
        ct_i = ct_i * cosd(yaw_angle_i)
        axial_ind_i = 0.5 / cosd(yaw_angle_i) * (1 - np.sqrt(1 - ct_i * cosd(yaw_angle_i)))
        turb_intensity_i = turbine_flow.turbulence_intensity_field_sorted_avg[i:i+1]
        rotor_diameter_i = np.ones_like(ct_i) * turbine_i.rotor_diameter
        hub_height_i = np.ones_like(ct_i) * turbine_i.hub_height

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        deflection_field = model.deflection.compute(
            x_i,
            y_i,
            effective_yaw_i,
            turb_intensity_i,
            ct_i,
            rotor_diameter_i,
            **deflection_args,
        )

        velocity_deficit = model.velocity.compute(
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

        wake_field = model.combination.compute(
            wake_field,
            velocity_deficit * flow.u_initial_sorted
        )

        flow.u_sorted = flow.u_initial_sorted - wake_field
        flow.v_sorted += v_wake
        flow.w_sorted += w_wake


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     TOOLS                                    #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def average_velocity(velocities,
                     ix_filter=None,
                     method='cubic-mean',
                     ) -> NDArrayFloat:

    if ix_filter is not None:
        velocities = velocities[ix_filter]
    if method == 'cubic-mean':
        return np.cbrt(np.mean(velocities ** 3.0, axis=(1, 2)))


def rotor_effective_velocity(air_density: float,
                             ref_density_cp_ct: float,
                             velocities: NDArrayFloat,
                             yaw_angle: NDArrayFloat,
                             pP: float,
                             pT: float,
                             turbine_type_map: NDArrayObject,
                             ix_filter: NDArrayInt | Iterable[int] | None = None,
                             average_method: str = "cubic-mean",
                             ) -> NDArrayFloat:

    if isinstance(yaw_angle, list):
        yaw_angle = np.array(yaw_angle)

    avg_vel = average_velocity(velocities, method=average_method)
    rotor_effective_velocities = (air_density/ref_density_cp_ct) ** (1/3) * avg_vel

    # Compute the rotor effective velocity adjusting for yaw settings
    rotor_effective_velocities = rotor_velocity_yaw_correction(
        pP, yaw_angle, rotor_effective_velocities
    )

    return rotor_effective_velocities


def rotor_velocity_yaw_correction(pP: float,
                                  yaw_angle: NDArrayFloat,
                                  rotor_effective_velocities: NDArrayFloat,
                                  ) -> NDArrayFloat:
    # Compute the rotor effective velocity adjusting for yaw settings
    pW = pP / 3.0  # Convert from pP to w
    return rotor_effective_velocities * cosd(yaw_angle) ** pW


def turbine_power_calculation(ref_density_cp_ct: float,
                              rotor_effective_velocities: NDArrayFloat,
                              power_interp: NDArrayObject,
                              turbine_type_map: NDArrayObject,
                              ix_filter: NDArrayInt | Iterable[int] | None = None,
                              ) -> NDArrayFloat:

    # Loop over each turbine type given to get power for all turbines
    # p = np.zeros(np.shape(rotor_effective_velocities))
    p = power_interp(rotor_effective_velocities)
    return p * ref_density_cp_ct


def get_plane_points(grid, flow, normal_vector, planar_coordinate=None):
    if (normal_vector == "z"):
            x_flat = grid.x_sorted_inertial_frame[0, 0].flatten()
            y_flat = grid.y_sorted_inertial_frame[0, 0].flatten()
            z_flat = grid.z_sorted_inertial_frame[0, 0].flatten()
    else:
        x_flat = grid.x_sorted[0, 0].flatten()
        y_flat = grid.y_sorted[0, 0].flatten()
        z_flat = grid.z_sorted[0, 0].flatten()

    u_flat = flow.u_sorted[0, 0].flatten()
    v_flat = flow.v_sorted[0, 0].flatten()
    w_flat = flow.w_sorted[0, 0].flatten()

    # Create a df of these
    if normal_vector == "z":
        df = pd.DataFrame(
            {
                "x1": x_flat,
                "x2": y_flat,
                "x3": z_flat,
                "u": u_flat,
                "v": v_flat,
                "w": w_flat,
            }
        )
    if normal_vector == "x":
        df = pd.DataFrame(
            {
                "x1": y_flat,
                "x2": z_flat,
                "x3": x_flat,
                "u": u_flat,
                "v": v_flat,
                "w": w_flat,
            }
        )
    if normal_vector == "y":
        df = pd.DataFrame(
            {
                "x1": x_flat,
                "x2": z_flat,
                "x3": y_flat,
                "u": u_flat,
                "v": v_flat,
                "w": w_flat,
            }
        )

    # Subset to plane
    if planar_coordinate is not None:
        df = df[np.isclose(df.x3, planar_coordinate)]  # , atol=0.1, rtol=0.0)]

    # Drop duplicates
    df = df.drop_duplicates()

    # Sort values of df to make sure plotting is acceptable
    df = df.sort_values(["x2", "x1"]).reset_index(drop=True)

    return df




if __name__ == '__main__':
    farm = WakeSimulator()
    # print(farm.turbine_power)
    farm.wake_field()