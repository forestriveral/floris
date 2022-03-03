import numpy as np
import pandas as pd
from scipy import integrate


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class Horns(object):
    @classmethod
    def layout(cls):
        c_n, r_n = 8, 10
        labels = []
        for i in range(1, r_n + 1):
            for j in range(1, c_n + 1):
                l = "c{}_r{}".format(j, i)
                labels.append(l)
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
        params["D_r"] = [80.] # 制动盘直径
        params["z_hub"] = [70.] # 轮毂高度
        params["v_in"] = [4.] # 切入风速
        params["v_rated"] = [15.] # 额定风速
        params["v_out"] = [25.] # 切出风速
        params["P_rated"] = [2.]  # 额定功率2MW
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
        return np.array([-2.98723724e-11, 5.03056185e-09,
                         -3.78603307e-07,  1.68050026e-05,
                         -4.88921388e-04,  9.80076811e-03,
                         -1.38497930e-01,  1.38736280e+00,
                         -9.76054549e+00,  4.69713775e+01,
                         -1.46641177e+02,  2.66548591e+02,
                         -2.12536408e+02]).dot(
                             np.array([vel**12, vel**11,
                                       vel**10, vel**9,
                                       vel**8, vel**7,
                                       vel**6, vel**5,
                                       vel**4, vel**3,
                                       vel**2, vel, 1.]))


class YawedLayoutPower(object):

    default_config = {"theta": 0.,
                      "inflow": 8.0,
                      "turb": 0.077,
                      "param": "horns",
                      "wm": "Bastankhah",
                      "wsm": "SumSquares",
                      "tim": "Frandsen", }

    def __init__(self, configs=None, **kwargs):
        configs = configs or self.default_config
        self.config_reset(configs, **kwargs)

    def config_reset(self, configs, **kwargs):
        self.config = {**self.default_config, **configs}

    def initial(self, layout, **kwargs):
        self.params = self.config["param"]
        self.turb = self.config["turb"]
        self.wm = self.models("wm")
        self.wsm = self.models("wsm")
        self.tim = self.models("tim")
        self.layout = layout
        self.wtnum = layout.shape[0]
        self.yawed = kwargs.get("yawed",
                                np.ones((self.wtnum, 2)) * [0., 0.15])
        self.param = self.params_uniform(self.wtnum)

    def models(self, model):
        default_model = {"wm":BastankhahWake,
                         "wsm":Sum_Squares,
                         "tim":Frandsen,}
        return default_model[model]

    def params_uniform(self, num):
        params = Horns.params().values
        cols = Horns.params().columns
        return pd.DataFrame(np.repeat(params, num, axis=0), columns=cols)

    def yawed_power(self, layouts, yaweds, configs=None, **kwargs):
        if configs is not None:
            self.config_reset(configs, **kwargs)
        self.initial(layouts, yawed=yaweds)
        powers = np.vectorize(Horns.pow_curve)(self.single_yawed)
        return powers

    @property
    def single_yawed(self):
        wt_loc = coordinate_transform(self.layout, self.config['theta'])
        wt_index = np.argsort(wt_loc[:, 1])[::-1]
        assert len(wt_index) == wt_loc.shape[0]
        deficits = np.zeros(len(wt_index))
        deficit_tab = np.full((len(wt_index), len(wt_index) + 2), None)
        turbulence_tab = np.full((len(wt_index), len(wt_index) + 2), None)
        for i, t in enumerate(wt_index):
            if i == 0:
                deficit_tab[0, -2], deficit_tab[0, -1] = 0., float(self.config["inflow"])
                if self.config["tim"] is not None:
                    turbulence_tab[0, -2], turbulence_tab[0, -1] = 0., self.config["turb"]
            ct_t, ytheta = 4 * self.yawed[t, 1] * (1 - self.yawed[t, 1]), self.yawed[t, 0]
            wake = self.wm(wt_loc[t, :], ct_t, self.param.iloc[t]["D_r"], self.param.iloc[t]["z_hub"],
                           I_a=self.config["turb"], ytheta=ytheta, T_m=self.tim,
                           I_w=turbulence_tab[i, -1],)
            if i < len(wt_index) - 1:
                for j, wt in enumerate(wt_index[i+1:]):
                    deficit_tab[i, i + j + 1], turbulence_tab[i, i + j + 1] = \
                        wake.wake_loss(wt_loc[wt, :], self.param.iloc[wt]["D_r"])
                total_deficit = self.wsm(deficit_tab[:, :], i + 1,
                                         inflow=float(self.config["inflow"]))
                if self.config["tim"] is not None:
                    turbulence_tab[i + 1, -2] = np.max(turbulence_tab[:i+1, i+1])
                    turbulence_tab[i + 1, -1] = np.sqrt(
                        np.max(turbulence_tab[:i+1, i+1])**2 + self.config["turb"]**2)
                deficit_tab[i + 1, -2] = total_deficit
                deficit_tab[i + 1, -1] = float(self.config["inflow"]) * (1 - total_deficit)
            else:
                break
            deficits[:] = powers_recorder(wt_index, deficit_tab[:, -1])
        return deficits


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                  WAKE MODELS                                 #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class BastankhahWake(object):
    def __init__(self, loc, C_t, D_r, z_hub, I_a=0.077, ytheta=0.,
                 T_m=None, I_w=None,):
        self.ref_loc = loc  # (x_axis, y_axis)
        self.C_thrust = C_t
        self.d_rotor = D_r
        self.z_hub = z_hub
        self.I_a = I_a
        self.epsilon = 0.2 * np.sqrt(
            (1. + np.sqrt(1 - self.C_thrust)) / (2. * np.sqrt(1 - self.C_thrust)))

        self.T_m = T_m
        self.I_wake = None if T_m is None else I_w
        self.k_star = 0.033 if T_m is None else 0.3837 * I_w + 0.003678
        self.r_D_ex, self.x_D_ex = self.wake_exclusion
        self.ytheta = ytheta

    def wake_sigma_Dr(self, x_D):
        return self.k_star * x_D + self.epsilon

    def deficit_constant(self, sigma_Dr):
        flag = self.C_thrust / (8 * sigma_Dr**2)
        if flag >= 1:
            # forcely change A(0D) to 0.99
            flag = 0.9999
            x_D = (sigma_Dr - self.epsilon) / self.k_star
            self.epsilon = np.sqrt(self.C_thrust / (8 * flag))
            sigma_Dr = self.wake_sigma_Dr(x_D)
        A = 1. - np.sqrt(1. - flag)
        B = -0.5 / ((self.d_rotor**2) * (sigma_Dr**2))
        return A, B

    def wake_integrand(self, sigma_Dr, d_spanwise):
        A, B = self.deficit_constant(sigma_Dr)
        return lambda r, t: A * np.exp(
            B * ((r * np.cos(t) + d_spanwise)**2 + (r * np.sin(t))**2)) * r

    def wake_velocity(self, inflow, x, y, z):
        A, B = self.deficit_constant(self.wake_sigma_Dr(x / self.d_rotor))
        v_deficit = A * np.exp(B * ((z - self.z_hub)**2 + y**2))
        return inflow * (1 - v_deficit)

    @staticmethod
    def wake_intersection(d_spanwise, r_wake, down_d_rotor):
        return wake_overlap(d_spanwise, r_wake, down_d_rotor)

    @property
    def wake_exclusion(self, m=0.01, n=0.05):
        x_D = np.arange(2, 40, 1)
        A, B = np.vectorize(self.deficit_constant)(self.wake_sigma_Dr(x_D))
        C = np.where(np.log(m / A) > 0., 0., np.log(m / A))
        r_D_ex = np.max(np.sqrt(C / B) / self.d_rotor)
        x_D_ex = (np.sqrt(self.C_thrust / (8 * (1 - (1 - n)**2))) \
            - self.epsilon) / self.k_star
        return r_D_ex, x_D_ex

    def wake_offset(self, ytheta, distance):
        ytheta, distance = ytheta / 360 * 2 * np.pi, distance / self.d_rotor
        theta_func = lambda x_D: np.tan(
            np.cos(ytheta)**2 * np.sin(ytheta) * self.C_thrust * 0.5 * (1 + 0.09 * x_D)**-2)
        offset = integrate.quad(theta_func, 0, distance)[0] * self.d_rotor
        return offset

    def wake_loss(self, down_loc, down_d_rotor):
        assert self.ref_loc[1] >= down_loc[1], "Reference WT must be upstream downstream WT!"
        d_streamwise,  d_spanwise = \
            np.abs(self.ref_loc[1] - down_loc[1]), np.abs(self.ref_loc[0] - down_loc[0])
        if d_streamwise == 0.:
            return 0., 0.
        sigma_Dr = self.wake_sigma_Dr(d_streamwise / self.d_rotor)
        if (d_spanwise / self.d_rotor) < self.r_D_ex or \
            (d_streamwise / self.d_rotor) < self.x_D_ex:
            wake_offset = self.wake_offset(self.ytheta, d_streamwise)
            if self.ref_loc[0] - down_loc[0] == 0:
                d_spanwise = np.abs(wake_offset)
            elif self.ref_loc[0] - down_loc[0] > 0:
                if wake_offset >= 0:
                    d_spanwise = np.abs(d_spanwise + wake_offset)
                else:
                    d_spanwise = np.abs(np.abs(d_spanwise) - np.abs(wake_offset))
            else:
                if wake_offset >= 0:
                    d_spanwise = np.abs(np.abs(d_spanwise) - np.abs(wake_offset))
                else:
                    d_spanwise = np.abs(d_spanwise + wake_offset)
            integral_velocity, _ = integrate.dblquad(
                self.wake_integrand(sigma_Dr, d_spanwise),
                0, 2 * np.pi, lambda r: 0, lambda r: down_d_rotor / 2)
            intersect_ratio = self.wake_intersection(
                    d_spanwise, 4 * sigma_Dr * self.d_rotor * 0.5, down_d_rotor) \
                        if self.T_m is not None else 0.
            I_add = self.T_m(self.C_thrust, self.I_wake, d_streamwise / self.d_rotor) \
                if self.T_m is not None else 0.
            return integral_velocity / (0.25 * np.pi * down_d_rotor**2), I_add * intersect_ratio
        else:
            return 0., 0.


def Frandsen(C_t, I_0, x_D):
    K_n = 0.4
    I_add = np.sqrt(K_n * C_t) / x_D
    return I_add


def Sum_Squares(deficits, i, **kwargs):
    return np.sqrt(np.sum(deficits[:i, i]**2))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def powers_recorder(index, result):
    tmp = np.zeros(result.shape[0])
    for i, r in enumerate(result):
        tmp[index[i]] = r
    return tmp


def coordinate_transform(coordinates, angle):
    return np.dot(
        coordinates,
        np.array([[np.cos(angle * np.pi / 180),
                   np.sin(angle * np.pi / 180)],
                  [- np.sin(angle * np.pi / 180),
                   np.cos(angle * np.pi / 180)]]))


def wake_overlap(d_spanwise, r_wake, down_d_rotor):
    if d_spanwise <= r_wake - (down_d_rotor / 2):
            return 1.
    elif d_spanwise > r_wake - (down_d_rotor / 2) and d_spanwise < r_wake + (down_d_rotor / 2):
        theta_w = np.arccos(
            (r_wake**2 + d_spanwise**2 - (down_d_rotor / 2)**2) / (2 * r_wake * d_spanwise))
        theta_r = np.arccos(((down_d_rotor / 2)**2 + d_spanwise **
                                2 - r_wake**2) / (2 * (down_d_rotor / 2) * d_spanwise))
        A_overlap = r_wake**2 * (theta_w - (np.sin(2 * theta_w) / 2)) + (
            (down_d_rotor / 2)**2) * (theta_r - (np.sin(2 * theta_r) / 2))
        return A_overlap / (np.pi * (down_d_rotor / 2)**2)
    else:
        return 0.


def yawed_generator(wtnum, num, seed=1234):
    induction_range = (0.031, 0.276)
    # thrust_range = (0.12, 0.8)
    yawed_range = (-20, 20)
    np.random.seed(seed)
    data = np.zeros((num * wtnum, 2))
    yawed_data = np.random.randint(
        yawed_range[0], yawed_range[1], wtnum * num)
    induction_data = np.random.uniform(
        induction_range[0], induction_range[1], wtnum * num)
    data[:, 0], data[:, 1] = yawed_data, induction_data
    return data.reshape((num, wtnum, 2))


def layout_generator(col=2, row=3, space=5, D=80.):
    layouts = [
        [i * space * D, j * space * D] for i in range(col) for j in range(row)]
    return np.array(layouts)


if __name__ == "__main__":

    config = {"theta": 30.,
              "inflow": 10.,
              "turb": 0.077, }

    # turbine position: [[x1, y1], [x2, y2], ...]
    layout = layout_generator()
    # yaw status: [[yaw1, ind1], [yaw2, ind2], ...]
    yawed = yawed_generator(layout.shape[0], 1)[0]
    # turbine powers: [p1, p2, p3, ...]
    farm = YawedLayoutPower()
    powers = farm.yawed_power(layout, yawed, config)
    print("turbine power: ", powers)
    print("farm power: ", np.sum(powers))

