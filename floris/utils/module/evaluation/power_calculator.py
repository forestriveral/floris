from __future__ import annotations

import os
import copy
import fnmatch
from pathlib import Path
from collections import OrderedDict

# import typing
import yaml
import numpy as np
import import_string

# from typing import Any, Tuple
# from attrs import define, field
import matplotlib.pyplot as plt

from floris.utilities import load_yaml
from floris.simulation import Floris, turbine
from floris.logging_manager import LoggerBase
from floris.simulation.wake import MODEL_MAP
from floris.tools import FlorisInterface, visualize_cut_plane
from floris.type_dec import NDArrayFloat, FromDictMixin


from floris.utils.module.tools import power_calc_ops as power_ops
from floris.utils.module.tools.farm_layout_loader import WindFarmLayout as WFL
# from floris.utils.visualization import power_calc_plot


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
        return {t.strip('.yaml') for t in turbines}


class FarmPower(FlorisInterface):

    turbine_library = _turbine_library_loader()
    parameter_strings = ["velocity_param", "deflection_param", "turbulence_param"]
    model_strings = ["velocity_model", "deflection_model",
                     "deflection_model", "combination_model"]
    required_strings = parameter_strings + model_strings

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
            assert isinstance(wake, dict) and set(wake.keys()).issubset(
                set(self.required_strings))
            floris_dict = copy.deepcopy(self.floris.as_dict())
            for key, value in wake.items():
                if key in floris_dict['wake']['model_strings'].keys():
                    assert value in MODEL_MAP[key].keys()
                    floris_dict['wake']['model_strings'][key] = value
                elif key in self.parameter_strings:
                    key_name = key.split('_')[0] + '_model'
                    model_name = wake[key_name] if key_name in wake.keys() \
                        else self.origin_floris.wake.model_strings[key_name]
                    floris_dict['wake'][f"wake_{key}eters"].update({model_name: value})
                elif key in floris_dict['wake'].keys():
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
            assert isinstance(turbine, (str, list))
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
        if (wind is None) or (isinstance(wind, dict) | len(wind) == 0):
            pass
        elif isinstance(wind, dict) and len(wind) > 0:
            # assert set(wind.keys()).issubset(set(self.required_strings))
            self.reinitialize(**wind)
        else:
            raise ValueError('Wind input is not valid')
        return self.floris.flow_field.as_dict()

    def wind_condition_array(self,
                             direction: int | float | list[int] | list[float] | NDArrayFloat,
                             speed: float | list[float] | NDArrayFloat,
                             sector: int | float | None = None,
                             interval: int | float = 1.,):
        if isinstance(direction, (int, float)):
            if isinstance(sector, (int, float)):
                wd = np.arange(direction - sector, direction + sector + interval, interval)
            elif isinstance(sector, None):
                wd = np.array([direction])
            else:
                raise ValueError("sector must be either None or a number")
        elif isinstance(direction, (list, np.ndarray)):
            # When under a range of wind direction, sector is ignored.
            assert len(list(direction)) == 2, "The length of direction list must be 2!"
            wd = np.arange(list(direction)[0], list(direction)[1] + interval, interval)
        else:
            raise ValueError("Wrong type of direction!")
        if isinstance(speed, (float, list, np.ndarray)):
            ws = np.array([speed]) if isinstance(speed, float) else np.array(speed)
        else:
            raise ValueError("Wrong type of speed!")
        return {'wind_directions': wd, 'wind_speeds': ws}

    def wake_calculation(self, yaw_angles: NDArrayFloat | list[float] | None = None,
                         layout: str | None = None, turbine: str | list | None = None,
                         wind : dict | None = None, wake: dict | None = None,):
        self.calculation_flag = self.farm_init(layout, turbine, wind, wake)
        self.calculate_wake(yaw_angles=yaw_angles)
        self.calculation_flag = True

    def turbine_power(self, mean: bool = False):
        # Power unit = Megawatts
        if not self.calculation_flag:
            self.wake_calculation()
        if mean:
            return np.round(np.mean(self.get_turbine_powers(), axis=1) * 1e-6, 4)
        else:
            return np.round(self.get_turbine_powers() * 1e-6, 4)

    def farm_power(self, mean: bool = True):
        # Power unit = Megawatts
        return np.sum(self.turbine_power(mean=mean), axis=-1)

    def baseline(self, params_dict: dict | str | Path, **kwargs):
        config, _ = self.wake_params_reader(params_dict, **kwargs)
        paper_name = config['paper']
        paper = import_string(f'floris.utils.tools.paper_data:{paper_name}')
        self.baseline_data = paper.baseline(layout=self.floris.name,
                                            fig_id=config['fig_id'],
                                            direction=config['direction'],
                                            turbine=config['turbine_id'],
                                            sector=config['sector'],
                                            **kwargs)
        return self.baseline_data

    def evaluation(self, params_dict: dict | str | Path, **kwargs):
        assert params_dict is not None, 'Params_dict must be provided!'
        config, wake = self.wake_params_reader(params_dict, **kwargs)
        case_name = list(wake.keys())
        print(case_name)
        self.case_data['config'] = copy.deepcopy(config)
        if config.get('turbine_id', None):
            baseline_data = self.baseline(params_dict, **kwargs)
            wind_direction, wind_sector = baseline_data[0][0], None
        else:
            wind_direction, wind_sector = config['direction'], config['sector']
        wind_speed = config.get('inflow', None) or self.origin_floris.flow_field.wind_speeds
        wind_turbine = config.get('turbine_type', None) or self.origin_floris.farm.turbine_type
        for (case, wake_param) in wake.items():
            if wake_param.get('turbine_type', None):
                wind_turbine = wake_param.pop('turbine_type')
            if wake_param.get('sector', None):
                wind_sector = wake_param.pop('sector')
            if wake_param.get('inflow', None):
                wind_speed = wake_param.pop('inflow')
            assert set(wake_param.keys()).issubset(set(self.required_strings))
            wind_condition = self.wind_condition_array(wind_direction, wind_speed, wind_sector)
            self.wake_calculation(turbine=wind_turbine, wind=wind_condition, wake=wake_param)
            self.case_data[case] = {**wind_condition, **{'power': self.turbine_power()}}
        # print(self.case_data['Jensen']['power'].shape)
        return self.case_data

    def visualization(self, params_dict: dict | str | Path =None, **kwargs):
        if params_dict is not None:
            case_data, baseline_data = self.evaluation(params_dict, **kwargs), \
                self.baseline(params_dict, **kwargs)
        else:
            assert self.case_data and self.baseline_data, \
                'Evaluation data must be calculated first!'
            case_data, baseline_data = self.case_data, self.baseline_data
        plot_name = case_data['config']['paper'] + '_visual'
        eval_plot = import_string(f'floris.utils.visualization.evaluation:{plot_name}')
        eval_plot.show(eval_data=case_data, baseline_data=baseline_data, **kwargs)

    def velocity_plot(self,
                      layout: str | None = None, turbine: str | list | None = None,
                      wind : dict | None = None, wake: dict | None = None, ):
        self.farm_init(layout, turbine, wind, wake)
        plot_height, rotor_diameter = self.floris.flow_field.reference_wind_height, \
            self.floris.farm.rotor_diameters[0, 0, 0]
        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
        horizontal_plane = self.calculate_horizontal_plane(
            height=plot_height, x_resolution=300, y_resolution=300,)
        visualize_cut_plane(horizontal_plane, ax=ax, title=self.layout['name'],
                            minSpeed=None, maxSpeed=None, cmap="coolwarm",
                            levels=None, color_bar=False,)
        # plt.savefig("../../outputs/farm.png", format='png', dpi=200, bbox_inches='tight')
        plt.show()

    def turbulence_plot(self, ):
        pass

    def wake_params_output(self, params_dict, output_path):
        with open(output_path, "w+") as f:
            yaml.dump(params_dict, f, sort_keys=False, indent=2,
                      default_flow_style=False)

    def wake_params_reader(self, params_dict: dict | str | Path,):
        if isinstance(params_dict, dict):
            wake_config = copy.deepcopy(params_dict)
        if isinstance(params_dict, (str, Path)):
            wake_config = load_yaml(f'{params_dict}.yaml')
        else:
            raise ValueError("Invalid import wake parameters YAML!")
        wake_config = OrderedDict(wake_config)
        config, wake = wake_config.pop('config', None), wake_config
        assert config.get('layout', None) == self.floris.name, \
            f"The layout name {config.get('layout', None)} of " + \
            f"config file does not match the name of the farm {self.floris.name}"
        return config, wake



if __name__ == "__main__":
    wind = {'wind_directions': [270.0], 'wind_speeds': [8.0]}
    LP = FarmPower('Lillgrund', wind=wind)
    # print(LP.floris.wake)
    # LP.evaluation(params_dict='../../inputs/wake_params',)
    # LP.evaluation(paper='WP_2015', params_dict='../../inputs/wake_params',
    #               fig_id='Fig_6', direction=270, sector=[1, 5])
    LP.visualization(params_dict='../../inputs/wake_params',)
    # LP.velocity_plot(wind={'wind_directions': [42.], 'wind_speeds': [8.]},)


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
