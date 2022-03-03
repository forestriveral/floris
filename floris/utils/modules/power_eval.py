import numpy as np
import import_string
import matplotlib.pyplot as plt

import floris.tools as wfct
import floris.tools.cut_plane as cp
import floris.tools.wind_rose as rose
import floris.tools.power_rose as pr
from floris.simulation.input_reader import InputReader
from floris.tools.optimization.scipy.yaw_wind_rose import \
    YawOptimizationWindRose

from floris.utils.tools import operation as ops
from floris.utils.tools.layout_loader import WindFarmLayout as WFL
from floris.utils.visualization import evaluation as veval

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     EVALUATION                               #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class LayoutPower(object):
    def __init__(self, farm, wakes=None, turbines=None):
        self.fi = wfct.floris_interface.FlorisInterface(farm)
        self.turbine = self.turbine_loader(turbines) \
            if turbines is not None else {}
        self.layout = [self.fi.layout_x, self.fi.layout_y]

    def turbine_loader(self, turbine_list):
        # import the turbine parameters for the calaculation of multiple turbine types
        # turbine_list = ["Vesta_2MW", "NREL_5MW"]
        if turbine_list is None:
            pass
        return [ops.json_load(turbine_list[k]) for _, k in enumerate(turbine_list.keys())]

    def wake_loader(self, wake_list):
        pass

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
        wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
        wfct.visualization.plot_turbines(ax, self.fi.layout_x, self.fi.layout_y,
                                         self.fi.get_yaw_angles(), D)
        # plt.savefig("images/hr1_270.png", format='png', dpi=200, bbox_inches='tight')
        plt.show()

    def turbulence_plot(self, ):
        pass



if __name__ == "__main__":
    LP = LayoutPower('inputs/horns_1.json')
    LP.evaluation()