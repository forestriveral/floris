from __future__ import annotations

import os
import copy
import fnmatch
import typing
import yaml
import numpy as np
import import_string
from pathlib import Path
from typing import Any, Tuple
from attrs import define, field
import matplotlib.pyplot as plt
from collections import OrderedDict


from floris.simulation import Floris, turbine
from floris.tools import FlorisInterface
from floris.logging_manager import LoggerBase
from floris.type_dec import NDArrayFloat, FromDictMixin
from floris.simulation.wake import MODEL_MAP
from floris.utilities import load_yaml

from floris.utils.tools import operation as tops
from floris.utils.tools.layout_loader import WindFarmLayout as WFL
from floris.utils.visualization import evaluation as veval



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     EVALUATION                               #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def _turbine_library_loader():
        # turbine_library = ["vesta_2MW", "nrel_5MW"]
        # Get a list of available turbine types
        turbine_library = '../../../turbine_library'
        my_turbine_library = '../../inputs/turbines/'
        turbines = os.listdir(turbine_library) + \
                fnmatch.filter(os.listdir(my_turbine_library), '*.yaml')
        return set([t.strip('.yaml') for t in turbines])


class FarmPower(FlorisInterface):
    turbine_library = _turbine_library_loader()
    def __init__(self,
                 config_file: dict | str | Path,
                 layout: str | None = None,
                 turbine: str | list | None = None,
                 wind : dict | None = None,
                 wake: dict | None = None,
                 het_map = None,):
        configuration = f"../../inputs/farms/{config_file}.yaml"
        FlorisInterface.__init__(self, configuration, het_map=het_map)
        self.origin_floris = copy.deepcopy(self.floris)
        self.calculation_flag = self.farm_init(layout, turbine, wind, wake)
        self.case_data, self.baseline_data = OrderedDict(), None

    def farm_init(self,
                  layout: str | None = None,
                  turbine: str | list | None = None,
                  wind : dict | None = None,
                  wake: dict | None = None,):
        self.floris = copy.deepcopy(self.origin_floris)
        self.layout = self.layout_init(layout)
        self.turbine = self.turbine_init(turbine)
        self.wind = self.wind_init(wind)
        self.wake = self.wake_init(wake)
        return False

    def wake_init(self, wake: dict | None):
        if wake is not None:
            # required_strings = self.floris.wake.model_strings.keys()
            wake_parameters = ["velocity_param", "deflection_param", "turbulence_param"]
            wake_models = ["velocity_model", "deflection_model", "deflection_model", "combination_model"]
            required_strings = wake_parameters + wake_models
            assert isinstance(wake, dict) and set(wake.keys()).issubset(set(required_strings))
            floris_dict = copy.deepcopy(self.floris.as_dict())
            for key, value in wake.items():
                if key in floris_dict['wake']['model_strings'].keys():
                    assert value in MODEL_MAP[key].keys()
                    floris_dict['wake']['model_strings'][key] = value
                elif key in wake_parameters:
                    key_name = key.split('_')[0] + '_model'
                    model_name = wake[key_name] if key_name in wake.keys() \
                        else self.origin_floris.wake.model_strings[key_name]
                    floris_dict['wake'][f"wake_{key}eters"].update({model_name: value})
                else:
                    if key in floris_dict['wake'].keys():
                        floris_dict['wake'][key] = value
                    else:
                        raise ValueError(f"{key} is not a valid wake parameter")
            self.floris = Floris.from_dict(floris_dict)
        return self.floris.wake.model_strings

    def layout_init(self, layout: str | None):
        layout = layout or WFL.layout(self.floris.name)
        self.reinitialize(layout=layout)
        self.turbine_num = self.floris.farm.n_turbines
        return {"name": self.floris.name,
                "Coordinate": np.array(self.get_turbine_layout()).T}

    def turbine_init(self, turbine: str | list | None):
        if turbine is not None:
            assert isinstance(turbine, str or list)
            turbine = [turbine] if isinstance(turbine, str) else turbine
            if len(turbine) not in (1, self.turbine_num):
                raise ValueError(
                f"The number of turbines in provided list ({len(turbine)}) " + \
                f"does not match the number of turbines in the farm ({self.turbine_num})"
                )
            assert set(turbine).issubset(self.turbine_library)
            self.reinitialize(turbine_type=turbine)
        return self.floris.farm.turbine_type

    def wind_init(self, wind: dict | None):
        self.reinitialize(**wind)
        return self.floris.flow_field.as_dict()

    def wind_direction_array(self,
                             direction: int | float | list[int] | list[float] | NDArrayFloat,
                             sector: int | float | None,
                             interval: int | float = 1.,):
        if isinstance(direction, int or float):
            if isinstance(sector, int or float):
                return np.arange(direction - sector, direction + sector + interval, interval)
            elif isinstance(sector, None):
                return np.array([direction])
            else:
                raise ValueError("sector must be either None or a number")
        elif isinstance(direction, list or NDArrayFloat):
            assert len(list(direction)) == 2, "The length of direction list must be 2!"
            return np.arange(list(direction)[0], list(direction)[1] + interval, interval)
        else:
            raise ValueError("Wrong type of direction!")

    def wake_calculation(self, yaw_angles: NDArrayFloat | list[float] | None = None,
                         wake: dict | None = None, layout: str | None = None,
                         turbine: str | list | None = None, wind : dict | None = None,):
        self.calculation_flag = self.farm_init(layout, turbine, wind, wake)
        self.calculate_wake(yaw_angles=yaw_angles)
        self.calculation_flag = True

    def turbine_power(self, mean: bool = False):
        # Power unit = Megawatts
        if not self.calculation_flag:
            self.wake_calculation()
        if mean:
            return np.round(np.mean(self.get_turbine_powers()[:, 0, :], axis=0) * 1e-6, 4)
        else:
            return np.round(self.get_turbine_powers()[:, 0, :] * 1e-6, 4)

    def farm_power(self, mean: bool = True):
        # Power unit = Megawatts
        return np.sum(self.turbine_power(mean=mean), axis=-1)

    @classmethod
    def paper_load(self, paper: str):
        return import_string(f'floris.utils.tools.paper_data:{paper}')

    def evaluation(self,
                   params_dict: dict | str | Path = None,
                   **kwargs):
        params_dict = OrderedDict(self.wake_params_reader(params_dict))
        config, wake = params_dict.pop('config', None), copy.deepcopy(params_dict)
        assert config.get('layout', None) == self.floris.name, \
            f"The layout name {config.get('layout', None)} of " + \
            f"config file does not match the name of the farm {self.floris.name}"
        self.baseline_data = self.paper_load(config['paper']).baseline(
            layout=self.floris.name, fig_id=config['fig_id'], direction=config['direction'],
            turbine=config['turbine_id'], sector=config['sector'], **kwargs)
        print(self.baseline_data[0], self.baseline_data[1])
        case_name = list(wake.keys())
        print(case_name)
        wind_direction, wind_sector = config['direction'], config['sector']
        wind_speed = config.get('inflow', None) or self.origin_floris.flow_field.wind_speeds
        wind_turbine = config.get('turbine_type', None) or self.origin_floris.farm.turbine_type
        for i, (case, wake_param) in enumerate(wake.items()):
            if wake_param.get('turbine_type', None):
                wind_turbine = wake_param.pop('turbine_type')
            if wake_param.get('sector', None):
                wind_sector = wake_param.pop('sector')
            if wake_param.get('inflow', None):
                wind_speed = wake_param.pop('inflow')
            direction_array = self.wind_direction_array(wind_direction, wind_sector)
            wind_condition = {'wind_directions': direction_array, 'wind_speeds': wind_speed}
            self.wake_calculation(turbine=wind_turbine, wind=wind_condition, wake=wake_param)
            self.case_data[case] = self.turbine_power()
        print(self.case_data['Jensen'].shape)
        return self.case_data, self.baseline_data

        

    def visualization(self, case_data, baseline_data):
        turbine_data, power_data = baseline_data[0], baseline_data[1]
        # array_power = veval.turbine_array_power(ref_inds, self.turbine_power[None, :])
        # veval.array_power_plot(array_power, labels=['270', ], ref=ref_data)
        plt.show()

    def velocity_plot(self, ax=None):
        H = self.fi.floris.farm.turbines[0].hub_height
        D = self.fi.floris.farm.turbine_map.turbines[0].rotor_diameter
        hor_plane = self.fi.get_hor_plane(height=H,
                                          x_resolution=500,
                                          y_resolution=500,)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        # wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
        # wfct.visualization.plot_turbines(ax, self.fi.layout_x, self.fi.layout_y,
        #                                  self.fi.get_yaw_angles(), D)
        # plt.savefig("images/hr1_270.png", format='png', dpi=200, bbox_inches='tight')
        plt.show()

    def turbulence_plot(self, ):
        pass

    @classmethod
    def wake_params_output(cls, params_dict, output_path):
        with open(output_path, "w+") as f:
            yaml.dump(params_dict, f, sort_keys=False, indent=2,
                      default_flow_style=False)

    @classmethod
    def wake_params_reader(cls, params_dict: dict | str | Path,):
        if isinstance(params_dict, dict):
            return copy.deepcopy(params_dict)
        if isinstance(params_dict, str or Path):
            return load_yaml(params_dict + '.yaml')
        else:
            raise ValueError("Invalid import wake parameters YAML!")



if __name__ == "__main__":
    wind = {'wind_directions': [270.0], 'wind_speeds': [8.0]}
    LP = FarmPower('Lillgrund', wind=wind)
    # print(LP.floris.wake)
    LP.evaluation(params_dict='../../inputs/wake_params',)
    # LP.evaluation(paper='WP_2015', params_dict='../../inputs/wake_params',
    #               fig_id='Fig_6', direction=270, sector=[1, 5])

    # wake_params = {"config": {"paper": "AV_2018",
    #                           "layout": "Lillgrund",
    #                           "fig_id": "Fig_4_6",
    #                           "direction": 42,
    #                           "turbine_id": None,
    #                           "sector": 10,
    #                           },
    #                "Jensen": {"velocity_model": "jensen",
    #                           "velocity_param": {"we": 0.05},
    #                           "deflection_model": "jimenez",
    #                           "deflection_param": {"ad": 0.0, "bd": 0.0, "kd": 0.05},
    #                           "turbine_type": "nrel_5MW"},
    #                "Gauss": {"velocity_model": "gauss",
    #                          "deflection_model": "gauss",
    #                          "turbine_type": "nrel_5MW"},
    #                }
    # FarmPower.wake_params_output(wake_params, '../../inputs/wake_params.yaml')
