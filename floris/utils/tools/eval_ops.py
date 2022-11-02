import copy
import math
import numpy as np
import pandas as pd

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                  VALIDATION                                  #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class CaseConfig(object):
    def __init__(self, configs):
        self.case_num = 1
        self.param_name = []
        self.case_list = None
        self.case_label = None
        self.label_ignore = None
        assert isinstance(configs, (pd.core.frame.DataFrame, dict)), \
            "Configuration info must be a DataFrame or dict object!"
        self.configs = self.unpack_config(configs) if isinstance(configs, dict) else configs

    def __getitem__(self, item):
        return self.configs[item] if type(item) == str else self.configs.iloc[:, item]

    @property
    def cases(self):
        return self.configs.columns

    @property
    def datatype(self):
        return type(self.configs)

    def unpack_config(self, config):
        conf = copy.deepcopy(config)
        ignore = conf.pop("ignore", [])
        self.label_ignore = np.full((len(conf)), False)
        self.case_list = np.zeros((1, len(conf)), dtype="O")
        for i, (t, p) in enumerate(conf.items()):
            self.param_name.append(t)
            if not isinstance(p, tuple) and (t in ignore):
                self.label_ignore[i] = True
            elif isinstance(p, tuple):
                self.label_ignore[2] = True
            p = self.parameters_format(p)
            self.case_list = np.tile(self.case_list, (len(p), 1))
            self.case_list[:, i] = np.repeat(np.array(p), self.case_num)
            self.case_num *= len(p)
        self.labels_format()
        assert self.case_list.shape[0] == len(self.case_label)
        return pd.DataFrame(self.case_list.T, columns=self.case_label,
                            index=self.param_name)

    def parameters_format(self, param):
        if (type(param) in [str, float, int]) or (param is None):
            return np.array([param], dtype="O")
        elif type(param) == tuple:
            x = np.zeros(1, dtype="O")
            x[0] = param
            return x
        elif type(param) == list:
            return np.array(param, dtype="O")
        else:
            raise ValueError("Invalid parameter format!")

    def labels_format(self):
        case = copy.deepcopy(self.case_list)
        self.case_label = np.zeros((case.shape[0]), dtype="O")
        if np.any(self.label_ignore):
            case[:, np.where(self.label_ignore == True)[0]] = \
                np.full((case.shape[0], len(np.where(self.label_ignore == True)[0])), None)
        for i in range(case.shape[0]):
            self.case_label[i] = "+".join(
                map(str, list(filter(None, list(case[i, :])))))


def coordinate_transform(coordinates, angle):
    return np.dot(coordinates, np.array([[np.cos(angle * np.pi / 180),
                                          np.sin(angle * np.pi / 180)],
                                         [- np.sin(angle * np.pi / 180),
                                          np.cos(angle * np.pi / 180)]]))


def wind_turbines_sort(coordinates):
    return np.argsort(coordinates[:, 1])[::-1]


def normalized_wf_power(powers):
    return np.sum(powers) / (np.max(powers) * powers.shape[0])


def normalized_wt_power(powers):
    return powers / np.max(powers)


def wt_power_reorder(index, result):
    tmp = np.zeros(result.shape[0])
    for i, r in enumerate(result):
        tmp[index[i]] = r
    return tmp


def target_power_extract(powers, targets):
    power_data = copy.deepcopy(powers)
    if targets.ndim != 2:
        return power_data[:, targets - 1]
    tmp = np.zeros(targets.shape)
    for j in range(power_data.shape[0]):
        for i in range(targets.shape[0]):
            tmp[i, :] = power_data[j, :][targets[i, :] - 1]
        power_data[j, :targets.shape[1]] = np.mean(tmp, axis=0)
    return power_data[:, :targets.shape[1]]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                  WAKE_MODELS                                 #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def wake_model_load(name, model):
    assert model in ["velocity", "combination", "turbulence", "cost"], \
        "Model type must be specific!"
    if name is None:
        return None
    from floris.utils.models.velocity import Bastankhah, Frandsen, Ishihara, \
        Jensen, Larsen, XieArcher
    from floris.utils.models.combination.VelocityCombination import Geometric_Sum, \
        Linear_Sum, Energy_Balance, Sum_Squares
    from floris.utils.models.turbulence.Uniform import Quart, Crespo, Frandsen_turb, \
        Larsen_turb, Tian, Gao, IEC
    from floris.utils.models.evaluation.EnergyCost import COE, LCOE

    VelocityModels = {'Bastankhah': Bastankhah.BastankhahWake,
                        'Frandsen': Frandsen.FrandsenWake,
                        'Ishihara': Ishihara.IshiharaWake,
                        'Jensen': Jensen.JensenWake,
                        'Larsen': Larsen.LarsenWake,
                        'XieArcher': XieArcher.XieArcherWake, }

    CombinationModels = {'GS': Geometric_Sum,
                            'LS': Linear_Sum,
                            'EB': Energy_Balance,
                            'SS': Sum_Squares, }

    TurbulenceModels = {'quart': Quart,
                        'Crespo': Crespo,
                        'Frandsen': Frandsen_turb,
                        'Larsen': Larsen_turb,
                        'tian': Tian,
                        'gao': Gao,
                        'IEC': IEC,
                        None: None}

    CostModels = {'COE': COE,
                    'LCOE': LCOE, }

    ModelsDict = {'velocity': VelocityModels,
                    'combination': CombinationModels,
                    'turbulence': TurbulenceModels,
                    'cost': CostModels, }

    return ModelsDict[model][name]


def wake_overlap(d_spanwise, r_wake, down_d_rotor):
    if d_spanwise <= r_wake - (down_d_rotor / 2):
        return 1.
    elif d_spanwise < r_wake + (down_d_rotor / 2):
        theta_w = np.arccos(
            (r_wake**2 + d_spanwise**2 - (down_d_rotor / 2)**2) / (2 * r_wake * d_spanwise))
        theta_r = np.arccos(((down_d_rotor / 2)**2 + d_spanwise **
                                2 - r_wake**2) / (2 * (down_d_rotor / 2) * d_spanwise))
        A_overlap = r_wake**2 * (theta_w - (np.sin(2 * theta_w) / 2)) + (
            (down_d_rotor / 2)**2) * (theta_r - (np.sin(2 * theta_r) / 2))
        return A_overlap / (np.pi * (down_d_rotor / 2)**2)
    else:
        return 0.


def wake_overlap_ellipse(d_spanwise, r_y_wake, r_z_wake, down_d_rotor):
    if d_spanwise <= r_y_wake - (down_d_rotor / 2):
        return 1.
    elif d_spanwise < r_y_wake + (down_d_rotor / 2):
        y_axis = quadratic_solver(1. - ( r_z_wake**2 / r_y_wake**2),
                                  - 2 * d_spanwise, d_spanwise**2 + r_z_wake**2 - 0.25 * down_d_rotor**2)[1]
        ellipse_radius = lambda y: np.sqrt(y**2 + (down_d_rotor / 2)**2 - (y - d_spanwise)**2)
        r_wake = ellipse_radius(y_axis)
        theta_w = np.arccos(
            (r_wake**2 + d_spanwise**2 - (down_d_rotor / 2)**2) / (2 * r_wake * d_spanwise))
        theta_r = np.arccos(((down_d_rotor / 2)**2 + d_spanwise **
                                2 - r_wake**2) / (2 * (down_d_rotor / 2) * d_spanwise))
        A_overlap = r_wake**2 * (theta_w - (np.sin(2 * theta_w) / 2)) + (
            (down_d_rotor / 2)**2) * (theta_r - (np.sin(2 * theta_r) / 2))
        return A_overlap / (np.pi * (down_d_rotor / 2)**2)
    else:
        return 0.


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


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def quadratic_solver(a, b, c):
    if a == 0:
        if b == 0:
            if c == 0:
                print('Countless solutions')
            else:
                print('Unsolvable')
        else:
            x = -c / b
            print('Solution: x1=%.3f' % x)
    else:
        q = (b ** 2) - (4 * a * c)
        if q > 0:
            x1 = (-b + math.sqrt(q)) / (a * 2)
            x2 = (-b - math.sqrt(q)) / (a * 2)
            print("Solution: x1=%.3f, x2=%.3f" % (x1, x2))
            return x1, x2
        elif q == 0:
            x1 = -b / a / 2
            x2 = x1
            print("Solution: x1=x2=%.3f" % (x1))
            return x1, x2
        else:
            print("Unsolvable")
            return None


def size_calculation():
    a = (np.tan(7 * np.pi/180) ** 2) + 1
    b = -14 * 80 * np.tan(7 * np.pi/180)
    c = 49 * 80 ** 2 - (9.3 * 80) ** 2

    aa, _ = quadratic_solver(a, b, c)
    bb = aa * np.tan(7 * np.pi / 180)
    cc = aa / np.cos(7 * np.pi / 180)

    print("a: %.3f | %.2fd" % (aa, aa / 80))
    print("b: %.3f | %.2fd" % (bb, bb / 80))
    print("c: %.3f | %.2fd" % (cc, cc / 80))

    # Solution: x1=558.616, x2=-423.140
    # a: 558.616 | 6.98d
    # b: 68.589 | 0.86d
    # c: 562.811 | 7.04d


# if __name__ == "__main__":

#     # random_curve()
#     pass

