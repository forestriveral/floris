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


from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from floris.logging_manager import LoggingManager
from floris.simulation import Floris, State
from floris.simulation.rotor_velocity import average_velocity
from floris.simulation.turbine.turbine import (
    axial_induction,
    power,
    thrust_coefficient,
)
from floris.tools.cut_plane import CutPlane
from floris.type_dec import NDArrayFloat


class FlorisInterface(LoggingManager):
    """
    FlorisInterface provides a high-level user interface to many of the
    underlying methods within the FLORIS framework. It is meant to act as a
    single entry-point for the majority of users, simplifying the calls to
    methods on objects within FLORIS.

    Args:
        configuration (:py:obj:`dict`): The Floris configuration dictionary or YAML file.
            The configuration should have the following inputs specified.
                - **flow_field**: See `floris.simulation.flow_field.FlowField` for more details.
                - **farm**: See `floris.simulation.farm.Farm` for more details.
                - **turbine**: See `floris.simulation.turbine.Turbine` for more details.
                - **wake**: See `floris.simulation.wake.WakeManager` for more details.
                - **logging**: See `floris.simulation.floris.Floris` for more details.
    """

    def __init__(self, configuration: dict | str | Path):
        self.configuration = configuration

        if isinstance(self.configuration, (str, Path)):
            try:
                self.floris = Floris.from_file(self.configuration)
            except FileNotFoundError:
                # If the file cannot be found, then attempt the configuration path relative to the
                # file location from which FlorisInterface was attempted to be run. If successful,
                # update self.configuration to an absolute, working file path and name.
                base_fn = Path(inspect.stack()[-1].filename).resolve().parent
                config = (base_fn / self.configuration).resolve()
                self.floris = Floris.from_file(config)
                self.configuration = config

        elif isinstance(self.configuration, dict):
            self.floris = Floris.from_dict(self.configuration)

        else:
            raise TypeError("The Floris `configuration` must be of type 'dict', 'str', or 'Path'.")

        # If ref height is -1, assign the hub height
        if np.abs(self.floris.flow_field.reference_wind_height + 1.0) < 1.0e-6:
            self.assign_hub_height_to_ref_height()

        # Make a check on reference height and provide a helpful warning
        unique_heights = np.unique(np.round(self.floris.farm.hub_heights, decimals=6))
        if ((
            len(unique_heights) == 1) and
            (np.abs(self.floris.flow_field.reference_wind_height - unique_heights[0]) > 1.0e-6
        )):
            err_msg = (
                "The only unique hub-height is not the equal to the specified reference "
                "wind height. If this was unintended use -1 as the reference hub height to "
                " indicate use of hub-height as reference wind height."
            )
            self.logger.warning(err_msg, stack_info=True)

        # Check the turbine_grid_points is reasonable
        if self.floris.solver["type"] == "turbine_grid":
            if self.floris.solver["turbine_grid_points"] > 3:
                self.logger.error(
                    f"turbine_grid_points value is {self.floris.solver['turbine_grid_points']} "
                    "which is larger than the recommended value of less than or equal to 3. "
                    "High amounts of turbine grid points reduce the computational performance "
                    "but have a small change on accuracy."
                )
                raise ValueError("turbine_grid_points must be less than or equal to 3.")

    def assign_hub_height_to_ref_height(self):

        # Confirm can do this operation
        unique_heights = np.unique(self.floris.farm.hub_heights)
        if len(unique_heights) > 1:
            raise ValueError(
                "To assign hub heights to reference height, can not have more than one "
                "specified height. "
                f"Current length is {unique_heights}."
            )

        self.floris.flow_field.reference_wind_height = unique_heights[0]

    def copy(self):
        """Create an independent copy of the current FlorisInterface object"""
        return FlorisInterface(self.floris.as_dict())

    def calculate_wake(
        self,
        yaw_angles: NDArrayFloat | list[float] | None = None,
        # tilt_angles: NDArrayFloat | list[float] | None = None,
    ) -> None:
        """
        Wrapper to the :py:meth:`~.Farm.set_yaw_angles` and
        :py:meth:`~.FlowField.calculate_wake` methods.

        Args:
            yaw_angles (NDArrayFloat | list[float] | None, optional): Turbine yaw angles.
                Defaults to None.
        """

        if yaw_angles is None:
            yaw_angles = np.zeros(
                (
                    self.floris.flow_field.n_findex,
                    self.floris.farm.n_turbines
                )
            )
        self.floris.farm.yaw_angles = yaw_angles

        # # TODO is this required?
        # if tilt_angles is not None:
        #     self.floris.farm.tilt_angles = tilt_angles
        # else:
        #     self.floris.farm.set_tilt_to_ref_tilt(
        #         self.floris.flow_field.n_findex,
        #     )

        # Initialize solution space
        self.floris.initialize_domain()

        # Perform the wake calculations
        self.floris.steady_state_atmospheric_condition()

    def calculate_no_wake(
        self,
        yaw_angles: NDArrayFloat | list[float] | None = None,
    ) -> None:
        """
        This function is similar to `calculate_wake()` except
        that it does not apply a wake model. That is, the wind
        farm is modeled as if there is no wake in the flow.
        Yaw angles are used to reduce the power and thrust of
        the turbine that is yawed.

        Args:
            yaw_angles (NDArrayFloat | list[float] | None, optional): Turbine yaw angles.
                Defaults to None.
        """

        if yaw_angles is None:
            yaw_angles = np.zeros(
                (
                    self.floris.flow_field.n_findex,
                    self.floris.farm.n_turbines
                )
            )
        self.floris.farm.yaw_angles = yaw_angles

        # Initialize solution space
        self.floris.initialize_domain()

        # Finalize values to user-supplied order
        self.floris.finalize()

    def reinitialize(
        self,
        wind_speeds: list[float] | NDArrayFloat | None = None,
        wind_directions: list[float] | NDArrayFloat | None = None,
        wind_shear: float | None = None,
        wind_veer: float | None = None,
        reference_wind_height: float | None = None,
        turbulence_intensity: float | None = None,
        # turbulence_kinetic_energy=None,
        air_density: float | None = None,
        # wake: WakeModelManager = None,
        layout_x: list[float] | NDArrayFloat | None = None,
        layout_y: list[float] | NDArrayFloat | None = None,
        turbine_type: list | None = None,
        turbine_library_path: str | Path | None = None,
        solver_settings: dict | None = None,
        heterogenous_inflow_config=None,
    ):
        # Export the floris object recursively as a dictionary
        floris_dict = self.floris.as_dict()
        flow_field_dict = floris_dict["flow_field"]
        farm_dict = floris_dict["farm"]

        # Make the given changes

        ## FlowField
        if wind_speeds is not None:
            flow_field_dict["wind_speeds"] = wind_speeds
        if wind_directions is not None:
            flow_field_dict["wind_directions"] = wind_directions
        if wind_shear is not None:
            flow_field_dict["wind_shear"] = wind_shear
        if wind_veer is not None:
            flow_field_dict["wind_veer"] = wind_veer
        if reference_wind_height is not None:
            flow_field_dict["reference_wind_height"] = reference_wind_height
        if turbulence_intensity is not None:
            flow_field_dict["turbulence_intensity"] = turbulence_intensity
        if air_density is not None:
            flow_field_dict["air_density"] = air_density
        if heterogenous_inflow_config is not None:
            flow_field_dict["heterogenous_inflow_config"] = heterogenous_inflow_config

        ## Farm
        if layout_x is not None:
            farm_dict["layout_x"] = layout_x
        if layout_y is not None:
            farm_dict["layout_y"] = layout_y
        if turbine_type is not None:
            farm_dict["turbine_type"] = turbine_type
        if turbine_library_path is not None:
            farm_dict["turbine_library_path"] = turbine_library_path

        ## Wake
        # if wake is not None:
        #     self.floris.wake = wake
        # if turbulence_kinetic_energy is not None:
        #     pass  # TODO: not needed until GCH

        if solver_settings is not None:
            floris_dict["solver"] = solver_settings

        floris_dict["flow_field"] = flow_field_dict
        floris_dict["farm"] = farm_dict

        # Create a new instance of floris and attach to self
        self.floris = Floris.from_dict(floris_dict)

    def get_plane_of_points(
        self,
        normal_vector="z",
        planar_coordinate=None,
    ):
        """
        Calculates velocity values through the
        :py:meth:`FlorisInterface.calculate_wake` method at points in plane
        specified by inputs.

        Args:
            normal_vector (string, optional): Vector normal to plane.
                Defaults to z.
            planar_coordinate (float, optional): Value of normal vector
                to slice through. Defaults to None.

        Returns:
            :py:class:`pandas.DataFrame`: containing values of x1, x2, x3, u, v, w
        """
        # Get results vectors
        if (normal_vector == "z"):
            x_flat = self.floris.grid.x_sorted_inertial_frame[0].flatten()
            y_flat = self.floris.grid.y_sorted_inertial_frame[0].flatten()
            z_flat = self.floris.grid.z_sorted_inertial_frame[0].flatten()
        else:
            x_flat = self.floris.grid.x_sorted[0].flatten()
            y_flat = self.floris.grid.y_sorted[0].flatten()
            z_flat = self.floris.grid.z_sorted[0].flatten()
        u_flat = self.floris.flow_field.u_sorted[0].flatten()
        v_flat = self.floris.flow_field.v_sorted[0].flatten()
        w_flat = self.floris.flow_field.w_sorted[0].flatten()

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
        # TODO: Seems sloppy as need more than one plane in the z-direction for GCH
        if planar_coordinate is not None:
            df = df[np.isclose(df.x3, planar_coordinate)]  # , atol=0.1, rtol=0.0)]

        # Drop duplicates
        # TODO is this still needed now that we setup a grid for just this plane?
        df = df.drop_duplicates()

        # Sort values of df to make sure plotting is acceptable
        df = df.sort_values(["x2", "x1"]).reset_index(drop=True)

        return df

    def calculate_horizontal_plane(
        self,
        height,
        x_resolution=200,
        y_resolution=200,
        x_bounds=None,
        y_bounds=None,
        wd=None,
        ws=None,
        yaw_angles=None,
    ):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a horizontal plane cut through
        the simulation domain at a specific height.

        Args:
            height (float): Height of cut plane. Defaults to Hub-height.
            x_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            y_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            y_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, y, u, v, w
        """
        # TODO update docstring
        if wd is None:
            wd = self.floris.flow_field.wind_directions
        if ws is None:
            ws = self.floris.flow_field.wind_speeds
        self.check_wind_condition_for_viz(wd=wd, ws=ws)

        # Store the current state for reinitialization
        floris_dict = self.floris.as_dict()
        current_yaw_angles = self.floris.farm.yaw_angles

        # Set the solver to a flow field planar grid
        solver_settings = {
            "type": "flow_field_planar_grid",
            "normal_vector": "z",
            "planar_coordinate": height,
            "flow_field_grid_points": [x_resolution, y_resolution],
            "flow_field_bounds": [x_bounds, y_bounds],
        }
        self.reinitialize(wind_directions=wd, wind_speeds=ws, solver_settings=solver_settings)

        # TODO this has to be done here as it seems to be lost with reinitialize
        if yaw_angles is not None:
            self.floris.farm.yaw_angles = yaw_angles

        # Calculate wake
        self.floris.solve_for_viz()

        # Get the points of data in a dataframe
        # TODO this just seems to be flattening and storing the data in a df; is this necessary?
        # It seems the biggest depenedcy is on CutPlane and the subsequent visualization tools.
        df = self.get_plane_of_points(
            normal_vector="z",
            planar_coordinate=height,
        )

        # Compute the cutplane
        horizontal_plane = CutPlane(
            df,
            self.floris.grid.grid_resolution[0],
            self.floris.grid.grid_resolution[1],
            "z"
        )

        # Reset the fi object back to the turbine grid configuration
        self.floris = Floris.from_dict(floris_dict)

        # Run the simulation again for futher postprocessing (i.e. now we can get farm power)
        self.calculate_wake(yaw_angles=current_yaw_angles)

        return horizontal_plane

    def calculate_cross_plane(
        self,
        downstream_dist,
        y_resolution=200,
        z_resolution=200,
        y_bounds=None,
        z_bounds=None,
        wd=None,
        ws=None,
        yaw_angles=None,
    ):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a horizontal plane cut through
        the simulation domain at a specific height.

        Args:
            height (float): Height of cut plane. Defaults to Hub-height.
            x_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            y_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            y_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, y, u, v, w
        """
        # TODO update docstring
        if wd is None:
            wd = self.floris.flow_field.wind_directions
        if ws is None:
            ws = self.floris.flow_field.wind_speeds
        self.check_wind_condition_for_viz(wd=wd, ws=ws)

        # Store the current state for reinitialization
        floris_dict = self.floris.as_dict()
        current_yaw_angles = self.floris.farm.yaw_angles

        # Set the solver to a flow field planar grid
        solver_settings = {
            "type": "flow_field_planar_grid",
            "normal_vector": "x",
            "planar_coordinate": downstream_dist,
            "flow_field_grid_points": [y_resolution, z_resolution],
            "flow_field_bounds": [y_bounds, z_bounds],
        }
        self.reinitialize(wind_directions=wd, wind_speeds=ws, solver_settings=solver_settings)

        # TODO this has to be done here as it seems to be lost with reinitialize
        if yaw_angles is not None:
            self.floris.farm.yaw_angles = yaw_angles

        # Calculate wake
        self.floris.solve_for_viz()

        # Get the points of data in a dataframe
        # TODO this just seems to be flattening and storing the data in a df; is this necessary?
        # It seems the biggest depenedcy is on CutPlane and the subsequent visualization tools.
        df = self.get_plane_of_points(
            normal_vector="x",
            planar_coordinate=downstream_dist,
        )

        # Compute the cutplane
        cross_plane = CutPlane(df, y_resolution, z_resolution, "x")

        # Reset the fi object back to the turbine grid configuration
        self.floris = Floris.from_dict(floris_dict)

        # Run the simulation again for futher postprocessing (i.e. now we can get farm power)
        self.calculate_wake(yaw_angles=current_yaw_angles)

        return cross_plane

    def calculate_y_plane(
        self,
        crossstream_dist,
        x_resolution=200,
        z_resolution=200,
        x_bounds=None,
        z_bounds=None,
        wd=None,
        ws=None,
        yaw_angles=None,
    ):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a horizontal plane cut through
        the simulation domain at a specific height.

        Args:
            height (float): Height of cut plane. Defaults to Hub-height.
            x_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            y_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            y_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, y, u, v, w
        """
        # TODO update docstring
        if wd is None:
            wd = self.floris.flow_field.wind_directions
        if ws is None:
            ws = self.floris.flow_field.wind_speeds
        self.check_wind_condition_for_viz(wd=wd, ws=ws)

        # Store the current state for reinitialization
        floris_dict = self.floris.as_dict()
        current_yaw_angles = self.floris.farm.yaw_angles

        # Set the solver to a flow field planar grid
        solver_settings = {
            "type": "flow_field_planar_grid",
            "normal_vector": "y",
            "planar_coordinate": crossstream_dist,
            "flow_field_grid_points": [x_resolution, z_resolution],
            "flow_field_bounds": [x_bounds, z_bounds],
        }
        self.reinitialize(wind_directions=wd, wind_speeds=ws, solver_settings=solver_settings)

        # TODO this has to be done here as it seems to be lost with reinitialize
        if yaw_angles is not None:
            self.floris.farm.yaw_angles = yaw_angles

        # Calculate wake
        self.floris.solve_for_viz()

        # Get the points of data in a dataframe
        # TODO this just seems to be flattening and storing the data in a df; is this necessary?
        # It seems the biggest depenedcy is on CutPlane and the subsequent visualization tools.
        df = self.get_plane_of_points(
            normal_vector="y",
            planar_coordinate=crossstream_dist,
        )

        # Compute the cutplane
        y_plane = CutPlane(df, x_resolution, z_resolution, "y")

        # Reset the fi object back to the turbine grid configuration
        self.floris = Floris.from_dict(floris_dict)

        # Run the simulation again for futher postprocessing (i.e. now we can get farm power)
        self.calculate_wake(yaw_angles=current_yaw_angles)

        return y_plane

    def check_wind_condition_for_viz(self, wd=None, ws=None):
        if len(wd) > 1 or len(wd) < 1:
            raise ValueError(
                "Wind direction input must be of length 1 for visualization. "
                f"Current length is {len(wd)}."
            )

        if len(ws) > 1 or len(ws) < 1:
            raise ValueError(
                "Wind speed input must be of length 1 for visualization. "
                f"Current length is {len(ws)}."
            )

    def get_turbine_powers(self) -> NDArrayFloat:
        """Calculates the power at each turbine in the wind farm.

        Returns:
            NDArrayFloat: Powers at each turbine.
        """

        # Confirm calculate wake has been run
        if self.floris.state is not State.USED:
            raise RuntimeError(
                "Can't run function `FlorisInterface.get_turbine_powers` without "
                "first running `FlorisInterface.calculate_wake`."
            )
        # Check for negative velocities, which could indicate bad model
        # parameters or turbines very closely spaced.
        if (self.floris.flow_field.u < 0.).any():
            self.logger.warning("Some velocities at the rotor are negative.")

        turbine_powers = power(
            velocities=self.floris.flow_field.u,
            air_density=self.floris.flow_field.air_density,
            power_functions=self.floris.farm.turbine_power_functions,
            yaw_angles=self.floris.farm.yaw_angles,
            tilt_angles=self.floris.farm.tilt_angles,
            tilt_interps=self.floris.farm.turbine_tilt_interps,
            turbine_type_map=self.floris.farm.turbine_type_map,
            turbine_power_thrust_tables=self.floris.farm.turbine_power_thrust_tables,
            correct_cp_ct_for_tilt=self.floris.farm.correct_cp_ct_for_tilt,
            multidim_condition=self.floris.flow_field.multidim_conditions
        )
        return turbine_powers

    def get_turbine_thrust_coefficients(self) -> NDArrayFloat:
        turbine_thrust_coefficients = thrust_coefficient(
            velocities=self.floris.flow_field.u,
            yaw_angles=self.floris.farm.yaw_angles,
            tilt_angles=self.floris.farm.tilt_angles,
            thrust_coefficient_functions=self.floris.farm.turbine_thrust_coefficient_functions,
            tilt_interps=self.floris.farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=self.floris.farm.correct_cp_ct_for_tilt,
            turbine_type_map=self.floris.farm.turbine_type_map,
            turbine_power_thrust_tables=self.floris.farm.turbine_power_thrust_tables,
            average_method=self.floris.grid.average_method,
            cubature_weights=self.floris.grid.cubature_weights,
            multidim_condition=self.floris.flow_field.multidim_conditions
        )
        return turbine_thrust_coefficients

    def get_turbine_ais(self) -> NDArrayFloat:
        turbine_ais = axial_induction(
            velocities=self.floris.flow_field.u,
            yaw_angles=self.floris.farm.yaw_angles,
            tilt_angles=self.floris.farm.tilt_angles,
            axial_induction_functions=self.floris.farm.turbine_axial_induction_functions,
            tilt_interps=self.floris.farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=self.floris.farm.correct_cp_ct_for_tilt,
            turbine_type_map=self.floris.farm.turbine_type_map,
            turbine_power_thrust_tables=self.floris.farm.turbine_power_thrust_tables,
            average_method=self.floris.grid.average_method,
            cubature_weights=self.floris.grid.cubature_weights,
            multidim_condition=self.floris.flow_field.multidim_conditions
        )
        return turbine_ais

    @property
    def turbine_average_velocities(self) -> NDArrayFloat:
        return average_velocity(
            velocities=self.floris.flow_field.u,
            method=self.floris.grid.average_method,
            cubature_weights=self.floris.grid.cubature_weights
        )

    def get_turbine_TIs(self) -> NDArrayFloat:
        return self.floris.flow_field.turbulence_intensity_field

    def get_farm_power(
        self,
        turbine_weights=None,
        use_turbulence_correction=False,
    ):
        """
        Report wind plant power from instance of floris. Optionally includes
        uncertainty in wind direction and yaw position when determining power.
        Uncertainty is included by computing the mean wind farm power for a
        distribution of wind direction and yaw position deviations from the
        original wind direction and yaw angles.

        Args:
            turbine_weights (NDArrayFloat | list[float] | None, optional):
                weighing terms that allow the user to emphasize power at
                particular turbines and/or completely ignore the power
                from other turbines. This is useful when, for example, you are
                modeling multiple wind farms in a single floris object. If you
                only want to calculate the power production for one of those
                farms and include the wake effects of the neighboring farms,
                you can set the turbine_weights for the neighboring farms'
                turbines to 0.0. The array of turbine powers from floris
                is multiplied with this array in the calculation of the
                objective function. If None, this  is an array with all values
                1.0 and with shape equal to (n_findex, n_turbines).
                Defaults to None.
            use_turbulence_correction: (bool, optional): When *True* uses a
                turbulence parameter to adjust power output calculations.
                Defaults to *False*.

        Returns:
            float: Sum of wind turbine powers in W.
        """
        # TODO: Turbulence correction used in the power calculation, but may not be in
        # the model yet
        # TODO: Turbines need a switch for using turbulence correction
        # TODO: Uncomment out the following two lines once the above are resolved
        # for turbine in self.floris.farm.turbines:
        #     turbine.use_turbulence_correction = use_turbulence_correction

        # Confirm calculate wake has been run
        if self.floris.state is not State.USED:
            raise RuntimeError(
                "Can't run function `FlorisInterface.get_turbine_powers` without "
                "first running `FlorisInterface.calculate_wake`."
            )

        if turbine_weights is None:
            # Default to equal weighing of all turbines when turbine_weights is None
            turbine_weights = np.ones(
                (
                    self.floris.flow_field.n_findex,
                    self.floris.farm.n_turbines
                )
            )
        elif len(np.shape(turbine_weights)) == 1:
            # Deal with situation when 1D array is provided
            turbine_weights = np.tile(
                turbine_weights,
                (
                    self.floris.flow_field.n_findex,
                    1
                )
            )

        # Calculate all turbine powers and apply weights
        turbine_powers = self.get_turbine_powers()
        turbine_powers = np.multiply(turbine_weights, turbine_powers)

        return np.sum(turbine_powers, axis=1)

    def get_farm_AEP(
        self,
        freq,
        cut_in_wind_speed=0.001,
        cut_out_wind_speed=None,
        yaw_angles=None,
        turbine_weights=None,
        no_wake=False,
    ) -> float:
        """
        Estimate annual energy production (AEP) for distributions of wind speed, wind
        direction, frequency of occurrence, and yaw offset.

        Args:
            freq (NDArrayFloat): NumPy array with shape (n_findex)
                with the frequencies of each wind direction and
                wind speed combination. These frequencies should typically sum
                up to 1.0 and are used to weigh the wind farm power for every
                condition in calculating the wind farm's AEP.
            cut_in_wind_speed (float, optional): Wind speed in m/s below which
                any calculations are ignored and the wind farm is known to
                produce 0.0 W of power. Note that to prevent problems with the
                wake models at negative / zero wind speeds, this variable must
                always have a positive value. Defaults to 0.001 [m/s].
            cut_out_wind_speed (float, optional): Wind speed above which the
                wind farm is known to produce 0.0 W of power. If None is
                specified, will assume that the wind farm does not cut out
                at high wind speeds. Defaults to None.
            yaw_angles (NDArrayFloat | list[float] | None, optional):
                The relative turbine yaw angles in degrees. If None is
                specified, will assume that the turbine yaw angles are all
                zero degrees for all conditions. Defaults to None.
            turbine_weights (NDArrayFloat | list[float] | None, optional):
                weighing terms that allow the user to emphasize power at
                particular turbines and/or completely ignore the power
                from other turbines. This is useful when, for example, you are
                modeling multiple wind farms in a single floris object. If you
                only want to calculate the power production for one of those
                farms and include the wake effects of the neighboring farms,
                you can set the turbine_weights for the neighboring farms'
                turbines to 0.0. The array of turbine powers from floris
                is multiplied with this array in the calculation of the
                objective function. If None, this  is an array with all values
                1.0 and with shape equal to (n_findex,
                n_turbines). Defaults to None.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the wake to
                the flow field. This can be useful when quantifying the loss
                in AEP due to wakes. Defaults to *False*.

        Returns:
            float:
                The Annual Energy Production (AEP) for the wind farm in
                watt-hours.
        """

        # Verify dimensions of the variable "freq"
        if np.shape(freq)[0] != self.floris.flow_field.n_findex:
            raise UserWarning(
                "'freq' should be a one-dimensional array with dimensions (n_findex). "
                f"Given shape is {np.shape(freq)}"
            )

        # Check if frequency vector sums to 1.0. If not, raise a warning
        if np.abs(np.sum(freq) - 1.0) > 0.001:
            self.logger.warning(
                "WARNING: The frequency array provided to get_farm_AEP() "
                "does not sum to 1.0."
            )

        # Copy the full wind speed array from the floris object and initialize
        # the the farm_power variable as an empty array.
        wind_speeds = np.array(self.floris.flow_field.wind_speeds, copy=True)
        wind_directions = np.array(self.floris.flow_field.wind_directions, copy=True)
        farm_power = np.zeros(self.floris.flow_field.n_findex)

        # Determine which wind speeds we must evaluate
        conditions_to_evaluate = wind_speeds >= cut_in_wind_speed
        if cut_out_wind_speed is not None:
            conditions_to_evaluate = conditions_to_evaluate & (wind_speeds < cut_out_wind_speed)

        # Evaluate the conditions in floris
        if np.any(conditions_to_evaluate):
            wind_speeds_subset = wind_speeds[conditions_to_evaluate]
            wind_directions_subset = wind_directions[conditions_to_evaluate]
            yaw_angles_subset = None
            if yaw_angles is not None:
                yaw_angles_subset = yaw_angles[conditions_to_evaluate]
            self.reinitialize(
                wind_speeds=wind_speeds_subset,
                wind_directions=wind_directions_subset
            )
            if no_wake:
                self.calculate_no_wake(yaw_angles=yaw_angles_subset)
            else:
                self.calculate_wake(yaw_angles=yaw_angles_subset)
            farm_power[conditions_to_evaluate] = (
                self.get_farm_power(turbine_weights=turbine_weights)
            )

        # Finally, calculate AEP in GWh
        aep = np.sum(np.multiply(freq, farm_power) * 365 * 24)

        # Reset the FLORIS object to the full wind speed array
        self.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions)

        return aep

    def sample_flow_at_points(self, x: NDArrayFloat, y: NDArrayFloat, z: NDArrayFloat):
        """
        Extract the wind speed at points in the flow.

        Args:
            x (1DArrayFloat | list): x-locations of points where flow is desired.
            y (1DArrayFloat | list): y-locations of points where flow is desired.
            z (1DArrayFloat | list): z-locations of points where flow is desired.

        Returns:
            3DArrayFloat containing wind speed with dimensions
            (# of findex, # of sample points)
        """

        # Check that x, y, z are all the same length
        if not len(x) == len(y) == len(z):
            raise ValueError("x, y, and z must be the same size")

        return self.floris.solve_for_points(x, y, z)

    def sample_velocity_deficit_profiles(
            self,
            direction: str = 'cross-stream',
            downstream_dists: NDArrayFloat | list = None,
            profile_range: NDArrayFloat | list = None,
            resolution: int = 100,
            wind_direction: float = None,
            homogeneous_wind_speed: float = None,
            ref_rotor_diameter: float = None,
            x_start: float = 0.0,
            y_start: float = 0.0,
            reference_height: float = None,
    ) -> list[pd.DataFrame]:
        """
        Extract velocity deficit profiles at a set of downstream distances from a starting point
        (usually a turbine location). For each downstream distance, a profile is sampled along
        a line in either the cross-stream direction (x2) or the vertical direction (x3).
        Velocity deficit is here defined as (homogeneous_wind_speed - u)/homogeneous_wind_speed,
        where u is the wake velocity obtained when wind_shear = 0.0.

        Args:
            direction: At each downstream location, this is the direction in which to sample the
                profile. Either `cross-stream` or `vertical`.
            downstream_dists: A list/array of streamwise locations for where to sample the profiles.
                Default starting point is (0.0, 0.0, reference_height).
            profile_range: Determines the extent of the line along which the profiles are sampled.
                The range is defined about a point which lies some distance directly downstream of
                the starting point.
            resolution: Number of sample points in each profile.
            wind_direction: A single wind direction.
            homogeneous_wind_speed: A single wind speed. It is called homogeneous since 'wind_shear'
                is temporarily set to 0.0 in this method.
            ref_rotor_diameter: A reference rotor diameter which is used to normalize the
                coordinates.
            x_start: x-coordinate of starting point.
            y_start: y-coordinate of starting point.
            reference_height: If `direction` is cross-stream, then `reference_height` defines the
                height of the horizontal plane in which the velocity profiles are sampled.
                If `direction` is vertical, then the velocity is sampled along the vertical
                direction with the `profile_range` being relative to the `reference_height`.
        Returns:
            A list of pandas DataFrame objects where each DataFrame represents one velocity deficit
            profile.
        """

        if direction not in ['cross-stream', 'vertical']:
            raise ValueError("`direction` must be either `cross-stream` or `vertical`.")

        if ref_rotor_diameter is None:
            unique_rotor_diameters = np.unique(self.floris.farm.rotor_diameters)
            if len(unique_rotor_diameters) == 1:
                ref_rotor_diameter = unique_rotor_diameters[0]
            else:
                raise ValueError(
                    "Please provide a `ref_rotor_diameter`. This is needed to normalize the "
                    "coordinates. Could not select a value automatically since the number of "
                    "unique rotor diameters in the turbine layout is not 1. "
                    f"Found the following rotor diameters: {unique_rotor_diameters}."
                )

        if downstream_dists is None:
            downstream_dists = ref_rotor_diameter * np.array([3, 5, 7, 9])

        if profile_range is None:
            profile_range = ref_rotor_diameter * np.array([-2, 2])

        wind_directions_copy = np.array(self.floris.flow_field.wind_directions, copy=True)
        wind_speeds_copy = np.array(self.floris.flow_field.wind_speeds, copy=True)
        wind_shear_copy = self.floris.flow_field.wind_shear

        if wind_direction is None:
            if len(wind_directions_copy) == 1:
                wind_direction = wind_directions_copy[0]
            else:
                raise ValueError(
                    "Could not determine a wind direction for which to sample the velocity "
                    "profiles. Either provide a single `wind_direction` as an argument to this "
                    "method, or initialize the Floris object with a single wind direction."
                )

        if homogeneous_wind_speed is None:
            if len(wind_speeds_copy) == 1:
                homogeneous_wind_speed = wind_speeds_copy[0]
                self.logger.warning(
                    "`homogeneous_wind_speed` not provided. Setting it to the following wind speed "
                    f"found in the current flow field: {wind_speeds_copy[0]} m/s. Note that the "
                    "inflow is always homogeneous when calculating the velocity deficit profiles. "
                    "This is done by temporarily setting `wind_shear` to 0.0"
                )
            else:
                raise ValueError(
                    "Could not determine a wind speed for which to sample the velocity "
                    "profiles. Provide a single `homogeneous_wind_speed` to this method."
                )

        if reference_height is None:
            reference_height = self.floris.flow_field.reference_wind_height

        self.reinitialize(
            wind_directions=[wind_direction],
            wind_speeds=[homogeneous_wind_speed],
            wind_shear=0.0,
        )

        velocity_deficit_profiles = self.floris.solve_for_velocity_deficit_profiles(
            direction,
            downstream_dists,
            profile_range,
            resolution,
            homogeneous_wind_speed,
            ref_rotor_diameter,
            x_start,
            y_start,
            reference_height,
        )

        self.reinitialize(
            wind_directions=wind_directions_copy,
            wind_speeds=wind_speeds_copy,
            wind_shear=wind_shear_copy,
        )

        return velocity_deficit_profiles

    @property
    def layout_x(self):
        """
        Wind turbine coordinate information.

        Returns:
            np.array: Wind turbine x-coordinate.
        """
        return self.floris.farm.layout_x

    @property
    def layout_y(self):
        """
        Wind turbine coordinate information.

        Returns:
            np.array: Wind turbine y-coordinate.
        """
        return self.floris.farm.layout_y

    def get_turbine_layout(self, z=False):
        """
        Get turbine layout

        Args:
            z (bool): When *True*, return lists of x, y, and z coords,
            otherwise, return x and y only. Defaults to *False*.

        Returns:
            np.array: lists of x, y, and (optionally) z coordinates of
                each turbine
        """
        xcoords, ycoords, zcoords = self.floris.farm.coordinates.T
        if z:
            return xcoords, ycoords, zcoords
        else:
            return xcoords, ycoords
