from __future__ import annotations

import copy
import numpy as np
import import_string
from pathlib import Path
from typing import Any, Tuple
from attrs import define, field
import matplotlib.pyplot as plt

from floris.simulation import Floris
from floris.tools import FlorisInterface
from floris.logging_manager import LoggerBase

from floris.utils.tools import operation as ops
from floris.utils.tools.layout_loader import WindFarmLayout as WFL
from floris.utils.visualization import evaluation as veval

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     EVALUATION                               #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class FarmPower(FlorisInterface):
    def __init__(self,
                 config_file: dict | str | Path,
                 wake: dict | None = None,
                 layout: str | None = None,
                 turbine: list | None = None,
                 het_map = None,):
        configuration = f"../inputs/{config_file}.yaml"
        FlorisInterface.__init__(self, configuration, het_map=het_map)
        self.wake = self.wake_init(wake)
        self.layout = self.layout_init(config_file, layout)

    def wake_init(self, wake):
        if wake is not None:
            required_strings = self.floris.wake.model_strings.keys()
            assert isinstance(wake, dict) and wake.keys().issubset(required_strings)
            floris_dict = copy.deepcopy(self.floris.as_dict())
            for key, value in wake.items():
                floris_dict['wake']['model_strings'][key] = value
            self.floris = Floris.from_dict(floris_dict)
            return wake
        else:
            return self.floris.wake.model_strings

    def layout_init(self, config_file, layout):
        layout = layout or WFL.layout(Path(config_file).name)
        self.reinitialize(layout=layout)
        return layout

    def turbine_init(self, turbine_list):
        # import the turbine parameters for the calaculation of multiple turbine types
        # turbine_list = ["Vesta_2MW", "NREL_5MW"]
        if turbine_list is None:
            pass
        return [ops.json_load(turbine_list[k]) for _, k in enumerate(turbine_list.keys())]

    def refer_loader(self, ref_data):
        # data_dir = ref_data.split('.')[0]
        return import_string(f'floris.utils.tools.paper_data:{ref_data}')

    @property
    def farm_power(self):
        return np.round(np.array(self.fi.get_farm_power()) * 1e-6, 4)

    @property
    def turbine_power(self):
        return np.round(np.array(self.fi.get_turbine_power()) * 1e-6, 4)

    def evaluation(self, layout='HR1', refer='WP_2015.Fig_6',
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

    def wake_plot(self, ax=None):
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
    LP = FarmPower('farms/template')
    # LP.evaluation()