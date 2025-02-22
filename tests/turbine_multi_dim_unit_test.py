# Copyright 2023 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from floris.simulation import (
    Turbine,
)
from floris.simulation.turbine.turbine import (
    axial_induction,
    power,
    thrust_coefficient,
)
from tests.conftest import SampleInputs, WIND_SPEEDS


TEST_DATA = Path(__file__).resolve().parent.parent / "floris" / "turbine_library"
CSV_INPUT = TEST_DATA / "iea_15MW_multi_dim_Tp_Hs.csv"


# size 16 x 1 x 1 x 1
# 16 wind speed and wind direction combinations from conftest
WIND_CONDITION_BROADCAST = np.reshape(np.array(WIND_SPEEDS), (-1, 1, 1, 1))

INDEX_FILTER = [0, 2]

# NOTE: MultiDimensionalPowerThrustTable not used anywhere, so I'm commenting
# this out.

# def test_multi_dimensional_power_thrust_table():
#     turbine_data = SampleInputs().turbine_multi_dim
#     turbine_data["power_thrust_data_file"] = CSV_INPUT
#     df_data = pd.read_csv(turbine_data["power_thrust_data_file"])
#     flattened_dict = MultiDimensionalPowerThrustTable.from_dataframe(df_data)
#     flattened_dict_base = {
#         ('Tp', '2', 'Hs', '1'): [],
#         ('Tp', '2', 'Hs', '5'): [],
#         ('Tp', '4', 'Hs', '1'): [],
#         ('Tp', '4', 'Hs', '5'): [],
#     }
#     assert flattened_dict == flattened_dict_base

#     # Test for initialization errors
#     for el in ("ws", "Cp", "Ct"):
#         df_data = pd.read_csv(turbine_data["power_thrust_data_file"])
#         df = df_data.drop(el, axis=1)
#         with pytest.raises(ValueError):
#             MultiDimensionalPowerThrustTable.from_dataframe(df)


def test_turbine_init():
    turbine_data = SampleInputs().turbine_multi_dim
    turbine_data["power_thrust_table"]["power_thrust_data_file"] = CSV_INPUT
    turbine = Turbine.from_dict(turbine_data)
    condition = (2, 1)
    assert turbine.rotor_diameter == turbine_data["rotor_diameter"]
    assert turbine.hub_height == turbine_data["hub_height"]
    assert turbine.power_thrust_table[condition]["pP"] == turbine_data["power_thrust_table"]["pP"]
    assert turbine.power_thrust_table[condition]["pT"] == turbine_data["power_thrust_table"]["pT"]
    assert turbine.generator_efficiency == turbine_data["generator_efficiency"]

    assert isinstance(turbine.power_thrust_table, dict)
    assert callable(turbine.thrust_coefficient_function)
    assert callable(turbine.power_function)
    assert turbine.rotor_radius == turbine_data["rotor_diameter"] / 2.0


def test_ct():
    N_TURBINES = 4

    turbine_data = SampleInputs().turbine_multi_dim
    turbine_data["power_thrust_table"]["power_thrust_data_file"] = CSV_INPUT
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]
    condition = (2, 1)

    # Single turbine
    # yaw angle / fCt are (n wind direction, n wind speed, n turbine)
    wind_speed = 10.0
    thrust = thrust_coefficient(
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        yaw_angles=np.zeros((1, 1)),
        tilt_angles=np.ones((1, 1)) * 5.0,
        thrust_coefficient_functions={turbine.turbine_type: turbine.thrust_coefficient_function},
        tilt_interps={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False]]),
        turbine_type_map=turbine_type_map[:,0],
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
        multidim_condition=condition
    )

    np.testing.assert_allclose(thrust, np.array([[0.77853469]]))

    # Multiple turbines with index filter
    # 4 turbines with 3 x 3 grid arrays
    thrusts = thrust_coefficient(
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 16 x 4 x 3 x 3
        yaw_angles=np.zeros((1, N_TURBINES)),
        tilt_angles=np.ones((1, N_TURBINES)) * 5.0,
        thrust_coefficient_functions={turbine.turbine_type: turbine.thrust_coefficient_function},
        tilt_interps={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False] * N_TURBINES]),
        turbine_type_map=turbine_type_map,
        ix_filter=INDEX_FILTER,
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
        multidim_condition=condition
    )
    assert len(thrusts[0]) == len(INDEX_FILTER)

    thrusts_truth = np.array([
        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.6957943,  0.6957943 ],

        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.6957943,  0.6957943 ],

        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.6957943,  0.6957943 ],

        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.77853469, 0.77853469],
        [0.6957943,  0.6957943 ],
    ])
    np.testing.assert_allclose(thrusts, thrusts_truth)

def test_power():
    N_TURBINES = 4
    AIR_DENSITY = 1.225

    turbine_data = SampleInputs().turbine_multi_dim
    turbine_data["power_thrust_table"]["power_thrust_data_file"] = CSV_INPUT
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]
    condition = (2, 1)

    # Single turbine
    wind_speed = 10.0
    p = power(
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        air_density=AIR_DENSITY,
        power_functions={turbine.turbine_type: turbine.power_function},
        yaw_angles=np.zeros((1, 1)), # 1 findex, 1 turbine
        tilt_angles=turbine.power_thrust_table[condition]["ref_tilt"] * np.ones((1, 1)),
        tilt_interps={turbine.turbine_type: turbine.tilt_interp},
        turbine_type_map=turbine_type_map[:,0],
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
        multidim_condition=condition
    )

    power_truth = 3029825.10569982

    np.testing.assert_allclose(p, power_truth)

    # Multiple turbines with ix filter
    velocities = np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST
    p = power(
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 16 x 4 x 3 x 3
        air_density=AIR_DENSITY,
        power_functions={turbine.turbine_type: turbine.power_function},
        yaw_angles=np.zeros((1, N_TURBINES)),
        tilt_angles=np.ones((1, N_TURBINES)) * 5.0,
        tilt_interps={turbine.turbine_type: turbine.tilt_interp},
        turbine_type_map=turbine_type_map,
        ix_filter=INDEX_FILTER,
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
        multidim_condition=condition
    )
    assert len(p[0]) == len(INDEX_FILTER)

    power_truth = turbine.power_function(
        power_thrust_table=turbine.power_thrust_table[condition],
        velocities=velocities,
        air_density=AIR_DENSITY,
        yaw_angles=np.zeros((1, N_TURBINES)),
        tilt_angles=np.ones((1, N_TURBINES)) * 5.0,
        tilt_interp=turbine.tilt_interp,
    )
    np.testing.assert_allclose(p, power_truth[:, INDEX_FILTER[0]:INDEX_FILTER[1]])


def test_axial_induction():

    N_TURBINES = 4

    turbine_data = SampleInputs().turbine_multi_dim
    turbine_data["power_thrust_table"]["power_thrust_data_file"] = CSV_INPUT
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]
    condition = (2, 1)

    baseline_ai = 0.2646995

    # Single turbine
    wind_speed = 10.0
    ai = axial_induction(
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        yaw_angles=np.zeros((1, 1)),
        tilt_angles=np.ones((1, 1)) * 5.0,
        axial_induction_functions={turbine.turbine_type: turbine.axial_induction_function},
        tilt_interps={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False]]),
        turbine_type_map=turbine_type_map[0,0],
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
        multidim_condition=condition
    )
    np.testing.assert_allclose(ai, baseline_ai)

    # Multiple turbines with ix filter
    ai = axial_induction(
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 16 x 4 x 3 x 3
        yaw_angles=np.zeros((1, N_TURBINES)),
        tilt_angles=np.ones((1, N_TURBINES)) * 5.0,
        axial_induction_functions={turbine.turbine_type: turbine.axial_induction_function},
        tilt_interps={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False] * N_TURBINES]),
        turbine_type_map=turbine_type_map,
        ix_filter=INDEX_FILTER,
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
        multidim_condition=condition
    )

    assert len(ai[0]) == len(INDEX_FILTER)

    # Test the 10 m/s wind speed to use the same baseline as above
    np.testing.assert_allclose(ai[2], baseline_ai)


def test_asdict(sample_inputs_fixture: SampleInputs):

    turbine = Turbine.from_dict(sample_inputs_fixture.turbine)
    dict1 = turbine.as_dict()

    new_turb = Turbine.from_dict(dict1)
    dict2 = new_turb.as_dict()

    assert dict1 == dict2
