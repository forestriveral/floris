import os
import sys
root = os.path.dirname(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.append(root)
import numpy as np
import pandas as pd
import import_string


config = {
        "num": 9,
        "param": "horns",
        "inflow": 10.0,
        "theta": 0.,
        "turb": 0.077,
        "wind": "horns",
        "wm": "Jensen",
        "wsm": "SS",
        "tim": None,
        "tsm": None,}


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class Horns(object):
    
    @classmethod
    def layout(cls):
        # Wind turbines labelling
        c_n, r_n = 8, 10
        labels = []
        for i in range(1, r_n + 1):
            for j in range(1, c_n + 1):
                l = "c{}_r{}".format(j, i)
                labels.append(l)
        # Wind turbines location generating  wt_c1_r1 = (0., 4500.)
        locations = np.zeros((c_n * r_n, 2))
        num = 0
        for i in range(r_n):
            for j in range(c_n):
                loc_x = 0. + 68.589 * j + 7 * 80. * i
                loc_y = 3911. - j * 558.616
                locations[num, :] = [loc_x, loc_y]
                num += 1
        return np.array(locations)
    
    @classmethod
    def params(cls):
        params = dict()
        
        params["D_r"] = [80.]
        params["z_hub"] = [70.]
        params["v_in"] = [4.]
        params["v_rated"] = [15.]
        params["v_out"] = [25.]
        params["P_rated"] = [2.]  # 2WM
        params["power_curve"] = ["horns"]
        params["ct_curve"] = ["horns"]
        
        return pd.DataFrame(params)
    
    @classmethod
    def pow_curve(cls, vel):
        if vel <= 4.:
            return 0.
        elif vel >= 15.:
            return 2.
        else:
            return 1.45096246e-07 * vel**8 - 1.34886923e-05 * vel**7 + \
                5.23407966e-04 * vel**6 - 1.09843946e-02 * vel**5 + \
                    1.35266234e-01 * vel**4 - 9.95826651e-01 * vel**3 + \
                        4.29176920e+00 * vel**2 - 9.84035534e+00 * vel + \
                            9.14526132e+00

    @classmethod
    def ct_curve(cls, vel):
        if vel <= 10.:
            vel = 10.
        elif vel >= 20.:
            vel = 20.
        return np.array([-2.98723724e-11, 5.03056185e-09, -3.78603307e-07,  1.68050026e-05,
                            -4.88921388e-04,  9.80076811e-03, -1.38497930e-01,  1.38736280e+00,
                            -9.76054549e+00,  4.69713775e+01, -1.46641177e+02,  2.66548591e+02,
                            -2.12536408e+02]).dot(np.array([vel**12, vel**11, vel**10, vel**9,
                                                            vel**8, vel**7, vel**6, vel**5,
                                                            vel**4, vel**3, vel**2, vel, 1.]))


class YawedLayoutPower(object):
    def __init__(self, configs, **kwargs):
        self.config = configs
        self.params = configs["param"]
        self.turb = configs["turb"]
        
        self.windn = configs["wind"]
        self.wmn = configs["wm"]
        self.wm = self.models(configs["wm"], "wm")
        self.wsmn = configs["wsm"]
        self.wsm = self.models(configs["wsm"], "wsm")
        self.timn = configs["tim"]
        self.tim = self.models(configs["tim"], "tim")
    
    def initial(self, layout, **kwargs):
        self.layout = layout
        self.wtnum = layout.shape[0]
        self.yawed = kwargs.get("yawed", None)
        self.param = self.params_uniform(self.wtnum)
    
    def models(self, name, model):
        assert model in ["wm", "wsm", "tim", "tsm", "cost"], \
        "Model type must be specific!"
        if name is None:
            return None
        if model == "wm":
            return import_string(f"models.wakes.{name}:{name}Wake")
        elif model == "wsm":
            return import_string(f"models.multiwakes.WSM:automator")(name)
        elif model == "tim":
            return import_string(f"models.turbs.TIM:{name}")
        elif model == "tsm":
            return import_string(f"models.turbs.TSM:{name}")
        elif model == "cost":
            return import_string(f"utils.cost:{name}")
        else:
            raise("Wrong model type. Please check!")
    
    def params_uniform(self, num):
        params = Horns.params().values
        cols = Horns.params().columns
        return pd.DataFrame(np.repeat(params, num, axis=0), columns=cols)

    def yawed_power(self, layouts, yaweds):
        if self.config['num'] != layouts.shape[0]:
            self.config['num'] = layouts.shape[0]
        self.initial(layouts, yawed=yaweds)
        velocity = self.single_yawed()
        powers = np.vectorize(Horns.pow_curve)(velocity)
        return powers
    
    def yawed_generator(self, layouts, N=500):
        yawed_data = random_yawed_generator(layouts.shape[0], N)
        power_data = np.zeros((N, layouts.shape[0]))
        for i in range(N):
            power_data[i, :] = self.yawed_power(
                layouts, yawed_data[i, :, :])
        return yawed_data, power_data
    
    def yawed_data(self, dsave=None):
        layouts = yawed_layout_generator()
        yawed_data = {"yawed":[], "power":[]}
        for i, layout in enumerate(layouts):
            # print(f"layout {i} shape", layout.shape)
            yawed, power = self.yawed_generator(layout)
            if dsave:
                np.save(f"{dsave}/yawed_{i + 1}.npy", yawed)
                np.save(f"{dsave}/power_{i + 1}.npy", power)
                print(f"== Data Save Done! ({dsave}) ==")
            yawed_data["yawed"].append(yawed)
            yawed_data["power"].append(power)
        return yawed_data
    
    
    def single_yawed(self):
        config, params, yawed, layout = \
            self.config, self.param, self.yawed, self.layout
        wt_loc = coordinate_transform(layout, self.config['theta'])
        wt_index = np.argsort(wt_loc[:, 1])[::-1]
        assert len(wt_index) == wt_loc.shape[0]
        deficits = np.zeros(len(wt_index))
        deficit_tab = np.full((len(wt_index), len(wt_index) + 2), None)
        turbulence_tab = np.full((len(wt_index), len(wt_index) + 2), None)
        for i, t in enumerate(wt_index):
            if i == 0:
                deficit_tab[0, -2], deficit_tab[0, -1] = 0., float(config["inflow"])
                if config["tim"] is not None:
                    turbulence_tab[0, -2], turbulence_tab[0, -1] = 0., config["Iam"]
            ct_t, ytheta = 4 * yawed[t, 1] * (1 - yawed[t, 1]), yawed[t, 0]
            wake = self.wm(wt_loc[t, :], ct_t, params.iloc[t]["D_r"],
                           params.iloc[t]["z_hub"], T_m=config["tim"],
                           I_w=turbulence_tab[i, -1], I_a=config["turb"],
                           ytheta=ytheta)
            if i < len(wt_index) - 1:
                for j, wt in enumerate(wt_index[i+1:]):
                    deficit_tab[i, i + j + 1], turbulence_tab[i, i + j + 1] = \
                        wake.wake_loss(wt_loc[wt, :], params.iloc[wt]["D_r"], debug=None)
                total_deficit = self.wsm(deficit_tab[:, :], i + 1,
                                         inflow=float(config["inflow"]))
                if config["tim"] is not None:
                    turbulence_tab[i + 1, -2] = np.max(turbulence_tab[:i+1, i+1])
                    turbulence_tab[i + 1, -1] = np.sqrt(
                        np.max(turbulence_tab[:i+1, i+1])**2 + config["turb"]**2)
                deficit_tab[i + 1, -2] = total_deficit
                deficit_tab[i + 1, -1] = float(config["inflow"]) * (1 - total_deficit)
            else:
                break
            deficits[:] = wt_power_reorder(wt_index, deficit_tab[:, -1])
        return deficits


def coordinate_transform(coordinates, angle):
    return np.dot(coordinates, np.array([[np.cos(angle * np.pi / 180),
                                          np.sin(angle * np.pi / 180)],
                                         [- np.sin(angle * np.pi / 180),
                                          np.cos(angle * np.pi / 180)]]))


def wt_power_reorder(index, result):
    tmp = np.zeros(result.shape[0])
    for i, r in enumerate(result):
        tmp[index[i]] = r
    return tmp


def random_yawed_generator(dim, N, seed=1234):
    # induction_range = (0.031, 0.276)
    # thrust_range = (0.12, 0.8)
    # yawed_range = (-30, 30)
    
    np.random.seed(seed)
    data = np.zeros((N * dim, 2))
    yawed_data = np.random.randint(-30, 30, dim * N)
    induction_data = np.random.uniform(0.031, 0.276, dim * N)
    data[:, 0], data[:, 1] = yawed_data, induction_data
    
    return data.reshape((N, dim, 2))


def yawed_layout_generator(num=5, D=80):
    
    layout_1 = np.array([[800, 3040],
                        [800, 2480],
                        [800, 1920],
                        [800, 1360],
                        [800, 800]])
    
    layout_2 = np.array([[800, 3040],
                        [800, 2480],
                        [800, 1920],
                        [1200, 3040],
                        [1200, 2480],
                        [1200, 1920],
                        [1600, 3040],
                        [1600, 2480],
                        [1600, 1920]])
    
    layout_3 = np.concatenate([
        layout_1 + np.array([num * i * D, 0]) for i in range(num)], axis=0)
    
    return layout_1, layout_2, layout_3


def yawed_power_5(x1, x2, x3, x4, x5,
                x6, x7, x8, x9, x10):
    
    layout = yawed_layout_generator()[0]
    yawed = np.array([[x1, x2],
                      [x3, x4],
                      [x5, x6],
                      [x7, x8],
                      [x9, x10]])
    powers = YawedLayoutPower(config).yawed_power(layout, yawed)
    
    return np.sum(powers)


def yawed_power_9(x1, x2, x3, x4, x5,
                  x6, x7, x8, x9, x10,
                  x11, x12, x13, x14, x15,
                  x16, x17, x18):
    
    layout = yawed_layout_generator()[1]
    yawed = np.array([[x1, x2],
                      [x3, x4],
                      [x5, x6],
                      [x7, x8],
                      [x9, x10],
                      [x11, x12],
                      [x13, x14],
                      [x15, x16],
                      [x17, x18]])
    powers = YawedLayoutPower(config).yawed_power(layout, yawed)
    
    return np.sum(powers)


def yawed_power_25(x1, x2, x3, x4, x5,
                x6, x7, x8, x9, x10):
    
    layout = yawed_layout_generator()[2]
    yawed = np.array([[x1, x2],
                      [x3, x4],
                      [x5, x6],
                      [x7, x8],
                      [x9, x10]])
    powers = YawedLayoutPower(config).yawed_power(layout, yawed)
    
    return np.sum(powers)



if __name__ == "__main__":

    layout = yawed_layout_generator()[1]
    # yawed = np.array([[0., 0.1], [0., 0.1], [0., 0.1], [0., 0.1], [0., 0.1]])
    yawed = np.array([[0., 0.1], [0., 0.1], [0., 0.1], [0., 0.1], [0., 0.1],
                      [0., 0.1], [0., 0.1], [0., 0.1], [0., 0.1]])
    powers = YawedLayoutPower(config).yawed_power(layout, yawed)
    print(powers)
