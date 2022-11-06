import time
import itertools
import numpy as np
import pandas as pd
# import multiprocessing as mp
# from multiprocessing import Pool as ProcessPool

from floris.utils.modules.tools import power_calc_ops_old as power_ops
from floris.utils.modules.tools import paper_data_extractor as paper_data
from floris.utils.modules.tools import power_calc_plot_old as power_plot


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class HornsRev1(object):
    def __init__(self):
        self.layout, self.label = self.layout_init()
        self.turbine = self.turbine_init()
        self.powers = pd.DataFrame()
        self.farm_power = False
        # self.pool = ProcessPool(int(mp.cpu_count()) - 2)  # 设置池的大小

    def layout_init(self):
        # Wind turbines labelling
        c_n, r_n = 8, 10
        labels = []
        for i, j in itertools.product(range(1, r_n + 1), range(1, c_n + 1)):
            l = "c{}_r{}".format(j, i)
            labels.append(l)
        # Wind turbines location generating  wt_c1_r1 = (0., 4500.)
        locations = np.zeros((c_n * r_n, 2))
        num = 0
        for num, (i, j) in enumerate(itertools.product(range(r_n), range(c_n))):
            loc_x = 0. + 68.589 * j + 7 * 80. * i
            loc_y = 4500. - j * 558.616
            locations[num, :] = [loc_x, loc_y]
        return locations, np.array(labels)

    def turbine_transform(self, theta):
        return power_ops.coordinate_transform(self.layout, theta)

    def turbine_init(self):
        return np.tile(np.array([{"D_r": 80., "z_hub": 70.}]), (80, ))

    def model_init(self, name, model):
        return power_ops.wake_model_load(name, model)

    def turbine_power(self, vel):
        return 0.17819 * vel**5 - 6.5198 * vel**4 + 90.623 * vel**3 - \
            574.62 * vel**2 + 1727.2 * vel - 1975

    def turbine_thrust(self, vel):
        if vel <= 5.:
            vel == 5.
        return -0.005694 * vel**2 + 0.06203 * vel + 0.637

    def layout_show(self, theta=0, annotate=True):
        return power_plot.layout_plot(self.turbine_transform(theta), annotate)

    def single_power(self, config, direction=None):
        theta = direction or config["theta"]
        wt_loc = self.turbine_transform(theta)
        wt_index = power_ops.wind_turbines_sort(wt_loc)
        assert len(wt_index) == wt_loc.shape[0]
        turbine_deficit = np.full((len(wt_index), len(wt_index) + 2), None)
        turbine_turb = np.full((len(wt_index), len(wt_index) + 2), None)
        turbine_deficit[0, -2], turbine_deficit[0, -1] = 0., float(config["inflow"])
        turbine_turb[0, -2], turbine_turb[0, -1] = 0., config["turb"]
        wake_model = self.model_init(config["wm"], model="wm")
        comb_model = self.model_init(config["wsm"], model="wsm")
        turb_model = self.model_init(config["tim"], model="tim")
        for i, t in enumerate(wt_index):
            if i < len(wt_index) - 1:
                wake = wake_model(wt_loc[t, :], self.turbine_thrust(turbine_deficit[i, -1]),
                                  self.turbine[t]["D_r"], self.turbine[t]["z_hub"],
                                  I_a=config["turb"], T_m=turb_model, I_w=turbine_turb[i, -1],)
                for j, wt in enumerate(wt_index[i+1:]):
                    deficit, turbulence = wake.wake_loss(wt_loc[wt, :], self.turbine[wt]["D_r"])
                    turbine_deficit[i, i + j + 1], turbine_turb[i, i + j + 1] = deficit, turbulence
                total_deficit = comb_model(turbine_deficit, i + 1, inflow=float(config["inflow"]))
                turbine_turb[i + 1, -2] = np.max(turbine_turb[:i + 1, i + 1])
                turbine_turb[i + 1, -1] = np.sqrt(
                    np.max(turbine_turb[:i + 1, i + 1])**2 + config["turb"]**2)
                turbine_deficit[i + 1, -1] = float(config["inflow"]) * (1 - total_deficit)
                turbine_deficit[i + 1, -2] = total_deficit
        return self.turbine_power(power_ops.wt_power_reorder(wt_index, turbine_deficit[:, -1]))

    def centered_power(self, config):
        center, sector = float(config["theta"]), float(config["sector"])
        calculated_direction = np.arange(center - sector, center + sector + 1)
        wt_powers = np.zeros((len(calculated_direction), self.layout.shape[0]))
        for i, w in enumerate(calculated_direction):
            wt_powers[i, :] = self.single_power(config, direction=w)
        return np.mean(wt_powers, axis=0)

    def ranged_power(self, config, step=1):
        calculated_direction = range(int(config["theta"][0]), int(config["theta"][1]), step)
        wf_powers = np.zeros(int((config["theta"][1] - config["theta"][0]) / step))
        for i, theta_i in enumerate(calculated_direction):
            wf_powers[i] = power_ops.normalized_wf_power(self.single_power(config, direction=theta_i))
        return wf_powers

    def model_power(self, configs, output=None, verbose=True):
        assert isinstance(configs, (power_ops.CaseConfig, dict)), \
            "Configuration info must be a DataFrame or dict object!"
        configs = power_ops.CaseConfig(configs) if isinstance(configs, dict) else configs
        assert len(configs.cases) != 0, "Case parameters needed!"
        if len(self.powers.columns) != 0:
            self.powers = pd.DataFrame()
        for i, case in enumerate(configs.cases):
            if verbose:
                print(f"====> calculating Case No.{i + 1}......")
                print(configs[i], '\n')
            time_start = time.time()
            direction, sector = configs[i]["theta"], configs[i]["sector"]
            if not isinstance(direction, (tuple, list)):
                assert isinstance(direction, (float, int)), \
                    "Invalid wind direction parameters for centered power calculation!"
                powers = self.centered_power(configs[i])
            else:
                assert float(sector) == 0., \
                    "Wind sector must be ZERO for ranged power calculation !"
                powers = self.ranged_power(configs[i])
                self.farm_power = True
            cost_time = round(time.time() - time_start, 3)
            print(f"...running time of {case}: {cost_time} sec")
            self.powers[case] = powers
        if output is not None:
            self.powers.index = np.arange(1, len(self.powers) + 1)
            self.powers.to_csv(f"../outputs/{output}.csv")

    def power_plot(self, config, target=None, **kwargs):
        self.model_power(power_ops.CaseConfig(config) if isinstance(config, dict) else config)
        if not self.farm_power:
            assert target is not None and target.all(), \
                "The targets turbine inside wind farm need to be specified!"
            turbine_power = power_ops.target_power_extract(self.powers.values.T, target)
            power_plot.wt_power_eval(self.powers.columns, turbine_power, **kwargs)
        else:
            power_plot.wf_power_eval(self.powers.columns, self.powers.values.T, **kwargs)

    def WP2015_Fig_6(self, configs, psave=None, dsave=None, show=True):
        direction, sector = configs["theta"], configs["sector"]
        if isinstance(direction, (tuple, list)):
            assert len(direction) == 1, "The length of tuple or list must be 1!"
            direction = direction[0]
        assert direction in (270., 222., 312.), \
            "Only data of 270/222/312 wind direction available!"
        turbine_target, turbine_power = paper_data.WP_2015.Fig_6(int(direction), sector)
        self.power_plot(configs, target=turbine_target, ref=turbine_power,
                        psave=psave, dsave=dsave, show=show)




if __name__ == "__main__":
    config = {
        "inflow": 8.,
        "theta": 270.,
        "turb": 0.077,
        "sector": [1., ],
        "wm": ['Bastankhah'],
        # "wm": ['Jensen', 'Bastankhah', 'Frandsen'],
        "wsm": ['SS', ],
        "tim": ["Frandsen", ],
        # "tim": [None, ],
        "ignore": ['inflow', 'turb', 'wsm'],
    }

    Horns = HornsRev1()
    # Horns.layout_show(theta=0.)
    Horns.WP2015_Fig_6(config, psave=None, dsave=None)
