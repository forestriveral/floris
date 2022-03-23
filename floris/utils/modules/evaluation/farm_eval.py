from __future__ import annotations

import os
import copy
import fnmatch
import typing
import numpy as np
import import_string
from pathlib import Path
from typing import Any, Tuple
from attrs import define, field
import matplotlib.pyplot as plt

from floris.simulation import Floris
from floris.type_dec import NDArrayFloat
from floris.tools import FlorisInterface
from floris.logging_manager import LoggerBase

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
                 wake: dict | None = None,
                 layout: str | None = None,
                 turbine: str | list | None = None,
                 wind : dict | None = None,
                 het_map = None,):
        configuration = f"../../inputs/{config_file}.yaml"
        FlorisInterface.__init__(self, configuration, het_map=het_map)
        self.farm_init(wake, layout, turbine, wind)

    def farm_init(self,
                  wake: dict | None = None,
                  layout: str | None = None,
                  turbine: str | list | None = None,
                  wind : dict | None = None,):
        self.wake = self.wake_init(wake)
        self.layout = self.layout_init(layout)
        self.turbine = self.turbine_init(turbine)
        self.wind = self.wind_init(wind)
        self.calculation_flag = False

    def wake_init(self, wake: dict | None):
        if wake is not None:
            required_strings = self.floris.wake.model_strings.keys()
            assert isinstance(wake, dict) and wake.keys().issubset(required_strings)
            floris_dict = copy.deepcopy(self.floris.as_dict())
            for key, value in wake.items():
                floris_dict['wake']['model_strings'][key] = value
            self.floris = Floris.from_dict(floris_dict)
        return self.floris.wake.model_strings

    def layout_init(self, layout: str | None):
        layout = layout or WFL.layout(self.floris.name)
        self.reinitialize(layout=layout)
        self.turbine_num = self.floris.farm.n_turbines
        return np.array(self.get_turbine_layout()).T

    def turbine_init(self, turbine: str | list | None):
        if turbine is not None:
            assert isinstance(turbine, typing.Union(str, list))
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

    def wake_calculation(self, yaw_angles: NDArrayFloat | list[float] | None = None,):
        self.calculate_wake(yaw_angles=yaw_angles)
        self.calculation_flag = True

    def turbine_power(self, mean=True):
        # Power unit = Megawatts
        if not self.calculation_flag:
            self.wake_calculation()
        if mean:
            return np.round(np.mean(self.get_turbine_power()[:, 0, :], axis=0) * 1e-6, 4)
        else:
            return np.round(self.get_turbine_power()[:, 0, :] * 1e-6, 4)

    def farm_power(self, mean=True):
        # Power unit = Megawatts
        return np.sum(self.turbine_power(mean=mean), axis=-1)

    def refer_loader(self, paper_data):
        # data_dir = ref_data.split('.')[0]
        return import_string(f'floris.utils.tools.paper_data:{paper_data}')

    @classmethod
    def evaluation(self,
                   layout='HR1',
                   refer='WP_2015.Fig_6',
                   wd=None, ws=None, **args):
        self.layout = WFL.layout(layout)
        ref_inds, ref_data = self.refer_loader(refer)(args)
        if isinstance():
            self.fi.reinitialize_flow_field(layout_array=self.layout,
                                            wind_direction=[270.0],
                                            wind_speed=[8.0])
            self.fi.calculate_wake()
        array_power = veval.turbine_array_power(ref_inds, self.turbine_power[None, :])
        veval.array_power_plot(array_power, labels=['270', ], ref=ref_data)

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

    def turb_plot(self, ):
        pass


if __name__ == "__main__":
    wind = {'wind_directions': [270.0], 'wind_speeds': [8.0]}
    LP = FarmPower('farms/template', wind=wind)
    # print(LP.turbine)
    LP.evaluation()
