import time
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool

import numpy as np
import pandas as pd

from floris.utils.tools import eval_ops as eops
from floris.utils.tools import farm_config as fconfig
from floris.utils.visual import wflo_eval as vweval
from floris.utils.visual import wflo_opt as vwopt

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class LayoutPower(object):
    def __init__(self, configs, **kwargs):
        self.config = configs
        self.params = configs["param"]
        self.vbins = configs["vbins"]
        self.wbins = configs["wbins"]
        self.turb = configs["turb"]
        self.bins = (self.vbins, self.wbins)
        self.uniform = self.uniform_check(configs["param"])
        self.wd_cdf = eops.wind_speed_dist()[1]

        self.wind_name = configs["wind"]
        self.cost_name = configs["cost"]
        self.vel_model = self.model_load(configs["velocity"], "velocity")
        self.comb_model = self.model_load(configs["combination"], "combination")
        self.turb_model = self.model_load(configs["turbulence"], "turbulence")
        self.cost_model = self.model_load(configs["cost"], "cost")
        self.water_depth = configs["wdepth"]

        self.pool = ProcessPool(int(mp.cpu_count()))

    def initialize(self, layout, **kwargs):
        self.layout = layout
        self.wtnum = layout.shape[0]
        self.yawed = kwargs.get("yawed",np.tile(np.array([0., None]), (self.wtnum, 1)))
        if not kwargs.get("params", None):
            self.param = self.params_uniform(self.wtnum)
            self.pow_curve = eops.params_loader(self.param["power_curve"][0]).pow_curve
        else:
            self.param = self.params_nonuniform(kwargs["params"])
        self.speed = (np.min(self.param["v_in"]), np.max(self.param["v_out"]))
        self.v_bin, self.v_point, self.w_bin, self.w_point = \
            self.discretization(self.bins, self.speed)
        self.wind, self.wind_pdf = self.data_load('wind'), self.data_load('pdf').values
        assert self.wind.shape[0] == self.w_point.shape[0]
        self.capacity = self.param["P_rated"].values

    def uniform_check(self, params):
        if isinstance(params, list):
            return False if len(params) > 1 else True
        else:
            return True

    def layout_check(self):
        pass

    def params_uniform(self, num):
        params = eops.params_loader(self.params).params().values
        cols = eops.params_loader(self.params).params().columns
        return pd.DataFrame(np.repeat(params, num, axis=0), columns=cols)

    def params_nonuniform(self, params):   # TODO
        self.uniform = False
        return None

    def data_load(self, data):
        return eops.winds_loader(data, self.wind_name, self.bins, self.speed)

    def model_load(self, name, model):
        return eops.find_and_load_model(name, model)

    def discretization(self, bins, speeds):
        return eops.winds_discretization(bins, speeds)

    def unpack_nonuniform(self, ):
        pass

    def plot_layout(self, layout, theta=0, annotate=False):
        return vweval.layout_plot(
            eops.coordinate_transform(layout, theta), annotate)

    def wakes(self, mprocess=False):
        wd_num, ws_num = self.w_point.shape[0], self.v_point.shape[0]
        if mprocess:
            deficits_calculator = self.deficits
            args = list(zip(list(self.w_point), [self.v_point] * wd_num,
                            [self.velocity_model] * wd_num, [self.comb_model] * wd_num,
                            [self.turb_model] * wd_num, [self.turb] * wd_num,
                            [self.param] * wd_num, [self.layout] * wd_num,))
            result = self.pool.map_async(deficits_calculator, args); result.wait()
            wt_deficits = np.transpose(np.array(result.get()), (1, 2, 0))
        else:
            wt_deficits = np.zeros((wd_num, ws_num, self.wtnum))
            for i, wd in enumerate(self.w_point):
                wt_deficits[i, :, :] = self.deficits(wd, self.layout)
            wt_deficits = np.transpose(wt_deficits, (1, 2, 0))
        return wt_deficits

    def one_wakes(self, layout, theta, mprocess=False, **kwargs):
        self.initialize(layout, **kwargs)
        if mprocess:
            deficits_calculator = self.deficits
            args = list(zip([theta], [self.v_point], [self.vel_model], [self.comb_model],
                            [self.turb_model], [self.turb], [self.param], [self.layout]))
            result = self.pool.map_async(deficits_calculator, args); result.wait()
            return result
        else:
            return self.deficits(theta, self.layout)

    def powers(self, deficits, params, **kwargs):
        pow_curve = eops.params_loader(params["power_curve"]).pow_curve \
            if not self.uniform else self.pow_curve
        wt_power = np.vectorize(pow_curve)(self.v_point[:, None] * (1. - deficits))
        no_wake_wt_power = \
            np.vectorize(pow_curve)(self.v_point[:, None] * np.ones((deficits.shape)))
        wd_powers = np.zeros((2, self.w_point.shape[0]))
        if kwargs.get("wd_output", False):
            wds_fs = self.wind_pdf / self.wind.values[:, -1][None, :]
            wd_power, no_wake_wd_power = \
                np.sum(wt_power * wds_fs, axis=0), np.sum(no_wake_wt_power * wds_fs, axis=0)
            wd_powers = np.concatenate((wd_power[None, :], no_wake_wd_power[None, :]), axis=0)
        wt_power, no_wake_wt_power = \
            np.sum(wt_power * self.wind_pdf), np.sum(no_wake_wt_power * self.wind_pdf)
        return np.array([wt_power, no_wake_wt_power], dtype=np.float), wd_powers

    def process(self, deficits, **kwargs):
        assert deficits.shape == (self.v_point.shape[0], self.wtnum, self.w_point.shape[0])
        powers, wd_powers = \
            np.zeros((self.wtnum, 2)), np.zeros((self.wtnum, 2, self.w_point.shape[0]))
        for i in range(self.wtnum):
            powers[i, :], wd_powers[i, :, :] = \
                self.powers(deficits[:, i, :], self.param.iloc[i], **kwargs)
        return powers, np.sum(wd_powers, axis=0).transpose(1, 0)

    def run(self, layout, **kwargs):
        self.initialize(layout, **kwargs)
        powers, wd_powers = self.process(self.wakes(mprocess=True), **kwargs)
        cost = self.cost_model(layout, powers, self.capacity, wdepth=self.water_depth, **kwargs)
        return cost, powers, wd_powers

    def test(self, layout, baseline=None, verbose=True, **kwargs):
        start = time.time()
        cost, powers, wd_powers = self.run(layout, **kwargs)
        end = time.time()
        if verbose:
            power, no_wake_power = np.sum(powers, axis=0)
            cf, eff,  =  power * 100 / np.sum(self.capacity), power * 100 / no_wake_power
            print(f"Interactive time: {end - start:.3f} s")
            print(f"Optimal({self.cost_name}[€/MWh] / Power[MW] / No-wake[MW] / " +
                f"CF[%] / Eff[%] / Loss[%]):\n ==> {cost:.3f} / {power:.3f} / " +
                f"{no_wake_power:.3f} / {cf:.2f} / {eff:.2f} / {100. - eff:.2f}\n")
            if baseline is not None:
                bcost, bpowers, _ = self.run(baseline, **kwargs)
                bpower, bno_wake_power = np.sum(bpowers, axis=0)[0], np.sum(bpowers, axis=0)[1]
                bcf, beff =  bpower * 100 / np.sum(self.capacity), bpower * 100 / bno_wake_power
                print(f"Baseline({self.cost_name}[€/MWh] / Power[MW] / No-wake[MW] / " +
                    f"CF[%] / Eff[%] / Loss[%]):\n ==> {bcost:.3f} / {bpower:.3f} / " +
                    f"{bno_wake_power:.3f} / {bcf:.2f} / {beff:.2f} / {100. - beff:.2f}\n")
        if kwargs.get("wd_output", False):
            assert wd_powers.all() != 0.
            vwopt.wd_power_plot(self.w_point, wd_powers, self.capacity, **kwargs)
        if kwargs.get("wt_output", False):
            vwopt.wt_power_plot(powers, self.capacity, **kwargs)
        return cost, powers

    def deficits(self, theta, layout, *args):
        # theta, speeds, wm, wsm, tim, turb, params, layout = args
        wt_loc = eops.coordinate_transform(layout, theta)
        wt_index = eops.wind_turbines_sort(wt_loc)
        assert wt_index.shape[0] == wt_loc.shape[0]
        deficits = np.zeros((len(self.v_point), len(wt_index)))
        turbine_deficit = np.full((len(self.v_point), len(wt_index), len(wt_index) + 2), None)
        turbine_turb = np.full((len(self.v_point), len(wt_index), len(wt_index) + 2), None)
        v_start = time.time()
        for z, v_i in enumerate(self.v_point):
            turbine_deficit[z, 0, -2], turbine_deficit[z, 0, -1] = 0., v_i
            turbine_turb[z, 0, -2], turbine_turb[z, 0, -1] = 0., self.turb
            for i, t in enumerate(wt_index):
                yaw_t, induction_t =  self.yawed[t, 0], self.yawed[t, 1]
                yaw_reduction = np.cos(yaw_t * np.pi / 180)**(1.88 / 3.0)
                turbine_deficit[z, i, -1] = turbine_deficit[z, i, -1] * yaw_reduction
                thrust = 4 * induction_t * (1 - induction_t) if induction_t is not None else \
                    eops.params_loader(self.param.iloc[t]["ct_curve"]).ct_curve(turbine_deficit[z, i, -1])
                if i < len(wt_index) - 1:
                    wake = self.velocity_model(wt_loc[t, :], thrust, self.param.iloc[t]["D_r"],
                                               self.param.iloc[t]["z_hub"], T_m=self.turb_model,
                                               I_w=turbine_turb[z, i, -1], I_a=self.turb, ytheta=yaw_t)
                    for j, wt in enumerate(wt_index[i+1:]):
                        turbine_deficit[z, i, i + j + 1], turbine_turb[z, i, i + j + 1] = \
                            wake.wake_loss(wt_loc[wt, :], self.param.iloc[wt]["D_r"], debug=None)
                    total_deficit = self.comb_model(turbine_deficit[z, :, :], i + 1, inflow=v_i)
                    turbine_turb[z, i + 1, -2] = np.max(turbine_turb[z, :i+1, i+1])
                    turbine_turb[z, i + 1, -1] = np.sqrt(
                        np.max(turbine_turb[z, :i+1, i+1])**2 + self.turb**2)
                    turbine_deficit[z, i + 1, -2] = total_deficit
                    turbine_deficit[z, i + 1, -1] = v_i * (1 - total_deficit)
            deficits[z] = eops.wt_power_reorder(wt_index, turbine_deficit[z, :, -2])
        v_end = time.time()
        print(f"Wind: {theta}  |  Time: {v_end - v_start}")
        return deficits


def analysis(path="solution", baseline="horns", result=None, config=None, **kwargs):
    result = result if isinstance(result, dict) else \
        eops.json_load(f"{path}/{result}.json")
    config = config or result['config']
    wf = LayoutPower(config)
    if wf.uniform:
        layout = np.array(result['layout'][-1]) if config['stage'] == 2 \
            else np.array(result['layout'])
        param = None
    else:
        print("NOTE: Nonuniform Wind Farm Configuration")
        layout, param = wf.unpack_nonuniform(result['layout'])
    # print(layout.shape)
    if layout.shape[0] == config['num']:
        wt_num = layout.shape[0]
    else:
        wt_num = layout.shape[0] // 2
        layout = layout.reshape((wt_num, 2))
    assert wt_num == config['num'], \
        'WTs number is not matching. Please check!'
    print("\nWind Turbine Num: ", wt_num)
    if baseline in ['horns', ]:
        baseline = eops.params_loader(baseline).baseline(wt_num)
    if (config["opt"] == "ga" and config["stage"] != 2) and config["grid"]:
        _, grids = eops.layout2grids([0, 0], [63, 48.89], config["grid"])
        layout = eops.grids2layout(layout, grids)
    layout = layout[np.argsort(layout[:, 1]), :] * 80.
    layout = layout - np.array([0, 589])
    cost, _ = wf.test(layout, baseline, param=param, path=path, **kwargs)
    if cost is not None:
        if kwargs.get("layout_output", False):
            vwopt.wf_layout_plot(layout, baseline, path=path, **kwargs)
        if kwargs.get("curve_output", False):
            vwopt.opt_curve_plot(result, path=path, **kwargs)



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #



if __name__ == "__main__":

    config = {
        "stage":2,
        "opt":['ga', 'pso'],
        "tag":'25',
        # "pop": 40,
        "pop": [20, 20],
        # "maxg": 5,
        "maxg": [20, 20],
        "grid": 5,
        "num": 25,
        "param": "horns",
        "wind": "horns",
        "vbins": 3,
        "wbins": 15,
        "wdepth": "linear_x",
        "cost": "LCOE",
        "turb": 0.077,
        "velocity": "Jensen",
        # "velocity": "Bastankhah",
        "combination": "SS",
        "turbulence": "Frandsen",
        # "turbulence": None,
    }

    layout = (fconfig.Horns.baseline(25) / 80.).ravel()
    # layout = None
    # path = "output/21_6_30/Jen_49_mos"
    # path = "solution"

    # analysis(path=path,
    #          baseline="horns",
    #          result="eapso_results_49",
    #         #  result={'layout': layout},
    #         #  config=config,
    #          layout_output=True,
    #          layout_name="layout_49",
    #          curve_output=True,
    #          curve_name="curve_25",
    #          wd_output=False,
    #          wd_name="wds_25",
    #          wt_output=False,
    #          wt_name="wts_25",
    #          )

    # LayoutPower(config).one_wakes(layout, 105., mprocess=True)

