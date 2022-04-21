import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geatpy as ea
from scipy import integrate

from floris.utils.tools import eval_ops as eops
from floris.utils.tools import farm_config as fconfig
from floris.utils.modules.optimization.wflo_layout import LayoutPower

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class WindFarm(object):
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


class YawedLayoutPower(object):#计算风场产能

    default_config = {"theta": 270.,
                      "inflow": 8.0,
                      "turb": 0.077,
                      "param": "horns",
                      "velocity": "Bastankhah",
                      "combination": "SumSquares",
                      "turbulence": "Frandsen", }

    def __init__(self, configs=None, **kwargs):
        configs = configs or self.default_config
        self.config_reset(configs, **kwargs)

    def config_reset(self, configs, **kwargs):
        self.config = {**self.default_config, **configs}

    def initial(self, layout, **kwargs):
        self.params = self.config["param"]
        self.turb = self.config["turb"]
        self.velocity = self.models("velocity")
        self.combination = self.models("combination")
        self.turbulence = self.models("turbulence")
        self.layout = layout
        self.wtnum = layout.shape[0]
        self.yawed = kwargs.get("yawed",
                                np.ones((self.wtnum, 2)) * [0., 0.15])
        self.param = self.params_uniform(self.wtnum)

    def models(self, model):
        default_model = {"velocity":BastankhahWake,
                         "combination":Sum_Squares,
                         "turbulence":Frandsen,}
        return default_model[model]

    def params_uniform(self, num):
        params = WindFarm.params().values
        cols = WindFarm.params().columns
        return pd.DataFrame(np.repeat(params, num, axis=0), columns=cols)

    def yawed_power(self, layouts, yaweds, configs=None, **kwargs):
        if configs is not None:
            self.config_reset(configs, **kwargs)
        self.initial(layouts, yawed=yaweds)
        powers = np.vectorize(WindFarm.pow_curve)(self.single_yawed)
        return powers

    @property
    def single_yawed(self):
        wt_loc = coordinate_transform(self.layout, self.config['theta'])
        wt_index = np.argsort(wt_loc[:, 1])[::-1]
        assert len(wt_index) == wt_loc.shape[0]
        turbine_deficit = np.full((len(wt_index), len(wt_index) + 2), None)
        turbine_turb = np.full((len(wt_index), len(wt_index) + 2), None)
        turbine_deficit[0, -2], turbine_deficit[0, -1] = 0., float(config["inflow"])
        turbine_turb[0, -2], turbine_turb[0, -1] = 0., config["turb"]
        for i, t in enumerate(wt_index):
            ct_t, ytheta = 4 * self.yawed[t, 1] * (1 - self.yawed[t, 1]), self.yawed[t, 0]
            if i < len(wt_index) - 1:
                wake = self.velocity(wt_loc[t, :], ct_t, self.param.iloc[t]["D_r"],
                                     self.param.iloc[t]["z_hub"], T_m=self.turbulence,
                                     I_w=turbine_turb[i, -1], I_a=self.config["turb"],
                                     ytheta=ytheta)
                for j, wt in enumerate(wt_index[i+1:]):
                    turbine_deficit[i, i + j + 1], turbine_turb[i, i + j + 1] = \
                        wake.wake_loss(wt_loc[wt, :], self.param.iloc[wt]["D_r"])
                total_deficit = self.combination(turbine_deficit[:, :], i + 1,
                                                 inflow=float(self.config["inflow"]))
                turbine_turb[i + 1, -2] = np.max(turbine_turb[:i + 1, i + 1])
                turbine_turb[i + 1, -1] = np.sqrt(
                    np.max(turbine_turb[:i + 1, i + 1])**2 + self.config["turb"]**2)
                turbine_deficit[i + 1, -1] = float(self.config["inflow"]) * (1 - total_deficit)
                turbine_deficit[i + 1, -2] = total_deficit
            yawed_power_reduction = np.cos(ytheta * np.pi / 180)**(1.88 / 3.0)
            turbine_deficit[i, -1] = turbine_deficit[i, -1] * yawed_power_reduction
        return powers_recorder(wt_index, turbine_deficit[:, -1])


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                  WAKE MODELS                                 #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class BastankhahWake(object):#BP尾流模型
    def __init__(self, loc, C_t, D_r, z_hub, I_a=0.077, ytheta=0.,
                 T_m=None, I_w=None,):#风机位置坐标；推力系数；风轮直径；轮毂高度；湍流度；ytheta？
        self.ref_loc = loc  # (x_axis, y_axis)
        self.C_thrust = C_t
        self.d_rotor = D_r
        self.z_hub = z_hub
        self.I_a = I_a
        self.epsilon = 0.2 * np.sqrt(
            (1. + np.sqrt(1 - self.C_thrust)) / (2. * np.sqrt(1 - self.C_thrust)))#

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
        v_deficit = A * np.exp(B * ((z - self.z_hub)**2 + y**2))#速度损失值
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

    def wake_offset(self, ytheta, distance):#偏航角
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


def Frandsen(C_t, I_0, x_D): ##湍流模型Frandsen
    K_n = 0.4
    I_add = np.sqrt(K_n * C_t) / x_D
    return I_add


def Sum_Squares(deficits, i, **kwargs):
    return np.sqrt(np.sum(deficits[:i, i]**2))#平方和尾流叠加模型


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


def wake_overlap(d_spanwise, r_wake, down_d_rotor):#尾流叠加模型
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


def yawed_generator(num, wtnum, seed=1234):
    induction_range = (0.031, 0.276)#诱导因子范围
    # thrust_range = (0.12, 0.8)#推力系数范围
    yawed_range = (-20, 20)#偏航角范围
    np.random.seed(seed)#生成一组随机数
    data = np.zeros((num * wtnum, 2))#生成两列wtnum * num行的空矩阵
    yawed_data = np.random.randint(
        yawed_range[0], yawed_range[1], wtnum * num)#在(-20, 20)范围内生成wtnum * num个随机整数
    induction_data = np.random.uniform(
        induction_range[0], induction_range[1], wtnum * num)#在[0.031, 0.276)范围内生成wtnum * num个随机数
    data[:, 0], data[:, 1] = yawed_data, induction_data
    return data.reshape((num, wtnum, 2))


def layout_generator(col=3, row=1, space=5, D=80.):
    layouts = [
        [i * space * D, j * space * D] for i in range(col) for j in range(row)]
    return np.array(layouts)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                  Fatigue Distribution                        #
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#


class FatigueYawedLayoutPower(YawedLayoutPower):  # 计算风场疲劳分布
    def __init__(self, configs=None, P_rate=5, T_life=6.3072e8,
                 C_rep=0.5, Lambda=0.5, **kwargs):
        super().__init__(configs=configs, **kwargs)
        self.P_rate = P_rate
        self.T_life = T_life
        self.C_rep = C_rep
        self.Lambda = Lambda

    def turbine_power(self, layouts, yaweds, configs=None, **kwargs):
        return self.yawed_power(layouts, yaweds, configs=configs, **kwargs)

    def turbine_fatigue(self, power, t=3.1536e7):
        fatigue = (1 + self.Lambda) * (power * t) / \
            self.P_rate * self.T_life * (1 + self.C_rep)
        return np.std(fatigue) / 1e15   # rescaling for large number


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                     Single-objective optimization                            #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class YawedPowerProblem(ea.Problem):
    def __init__(self, configs, layout):
        name = 'YawedPower'  # 初始化name（函数名称，可以随意设置）
        self.configs = configs
        self.layout = layout
        self.Dim = 2 * self.configs["num"]  # 初始化Dim（决策变量维数）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1] * M  # 初始化目标最小最大化标记列表，1:min；-1:max
        varTypes = [0] * self.Dim # 初始化决策变量类型，0:连续；1:离散

        lb = [-30, 0.031] * self.configs["num"]
        ub = [30, 0.276] * self.configs["num"]
        lbin = [0] * self.Dim  # 决策变量下边界
        ubin = [0] * self.Dim  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, self.Dim,
                            varTypes, lb, ub, lbin, ubin)
        self.calculator = YawedLayoutPower(self.configs)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        Vars = pop.Phen.reshape(pop.Phen.shape[0], self.configs["num"], 2)  # 得到决策变量矩阵
        pop.ObjV = np.zeros((Vars.shape[0], self.M))
        for i in range(Vars.shape[0]):
            powers = self.calculator.yawed_power(self.layout, Vars[i, :, :])
            pop.ObjV[i, :] = np.sum(powers)


class YawedOpt(object):
    def __init__(self, config, layout):
        self.config = config
        self.problem = YawedPowerProblem(config, layout) # 实例化问题对象
        """==============================种群设置==========================="""
        self.Encoding = 'RI' # 编码方式
        self.NIND = config["pop"] # 种群规模
        self.MAXGEN = config["maxg"]

        self.Field = ea.crtfld(self.Encoding, self.problem.varTypes, self.problem.ranges,
                               self.problem.borders) # 创建区域描述器
        self.population = ea.Population(self.Encoding, self.Field, self.NIND) # 实例化种群对象（此时种群还没被真正初始化，仅仅是生成一个种群对象）

    def solution(self, ):
    # 单目标带约束
        """===========================算法参数设置=========================="""
        # myAlgorithm = ea.soea_SEGA_templet(self.problem, self.population) # 实例化一个算法模板对象
        # myAlgorithm.mutOper.Pm = 0.5  # 变异概率
        myAlgorithm = ea.soea_DE_rand_1_bin_templet(self.problem, self.population)  # 实例化一个算法模板对象
        myAlgorithm.mutOper.F = 0.6 # 差分进化中的参数F
        myAlgorithm.recOper.XOVR = 0.5 # 设置交叉概率
        myAlgorithm.trappedValue = 1e-3  # “进化停滞”判断阈值
        myAlgorithm.maxTrappedCount = 50  # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化

        myAlgorithm.MAXGEN = self.MAXGEN # 最大进化代数
        myAlgorithm.logTras = 1 #设置每隔多少代记录日志，若设置成0则表示不记录日志
        myAlgorithm.verbose = True # 设置是否打印输出日志信息
        myAlgorithm.drawing = 1
        # 设置绘图方式（0:不绘图；1:绘制结果图；2:绘制目标空间过程动画； 3:绘制决策空间过程动画）
        """==========================调用算法模板进行种群进化==============="""
        [BestIndi, self.population] = myAlgorithm.run() # 执行算法模板，得到最优个体以及最后一代种群
        BestIndi.save("solution/Results") # 把最优个体的信息保存到文件中
        """=================================输出结果======================="""
        print('Evaluation times:{}'.format(myAlgorithm.evalsNum))
        print('Elapsed time %s %s' % eops.time_formator(myAlgorithm.passTime))
        if BestIndi.sizes != 0:
            yawed_data = pd.read_csv(
                f"../solution/Results/Phen.csv",
                header=None).values.reshape(self.config["num"], 2)
            print('Optimal Yawed:(Turbine: Yaw/Induction)')
            for i in range(yawed_data.shape[0]):
                print(f'   {i+1}: {yawed_data[i, 0]:.1f} / {yawed_data[i, 1]:.3f}')
            print('Optimal Power:%s' % (BestIndi.ObjV[0][0]))
        else:
            print('No feasible solution')


def yawed_case_run(N, pop=None, maxg=None):
    config = {
        "pop": pop or 20,
        "maxg": maxg or 50,
        "num": N,
        "param": "horns",
        "inflow": 8.0,
        "theta": 0.,
        "sector": 3,
        "Iam": 0.077,
        "winds": "horns",
        "velocity": "Jensen",
        "combination": "SS",
        "turbulence": None,
        "superposition": None,}
    layout_5, layout_9, layout_25 = eops.yawed_layout_generator()
    YawedOpt(config, eval(f"layout_{N}")).solution()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                    Multi-objective optimization                              #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ZDT1(ea.Problem): # 继承Problem父类

    default_config = {"theta": 270.,
                      "inflow": 12.0,
                      "turb": 0.077,
                      "param": "horns",
                      "velocity": "Bastankhah",
                      "combination": "SumSquares",
                      "turbulence": "Frandsen", }

    def __init__(self, configs=None, layout=None, **kwargs):
        self.configs = configs or self.default_config
        self.layout = layout_generator() if layout is None else layout
        self.wt_num = self.layout.shape[0]
        name = 'ZDT1'           # 初始化name（函数名称，可以随意设置）
        M = 2                   # 初始化M（目标维数）
        maxormins = [1] * M     # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
        # loading the turbine number from layout array or set it to default value (=4)
        Dim = 2 * self.wt_num   # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim    # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [-30, 0.031] * self.wt_num   # 决策变量下界
        ub = [30, 0.276] * self.wt_num    # 决策变量上界
        lbin = [1] * Dim        # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim        # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        self.calculator = FatigueYawedLayoutPower(self.configs, **kwargs)
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop=None):  # 目标函数
        yawVars = pop.Phen.reshape(pop.Phen.shape[0], -1, 2) if pop is not None \
            else yawed_generator(40, self.wt_num) # Obtain the decision variables of pop
        yawObjV = np.zeros((yawVars.shape[0], 2))  # Building the objective variable matrix
        for i in range(yawVars.shape[0]):
            powers = self.calculator.turbine_power(self.layout, yawVars[i, :, :])
            fatigue = self.calculator.turbine_fatigue(powers)
            yawObjV[i, 0], yawObjV[i, 1] = 1 / np.sum(powers), fatigue
        if pop is not None:
            pop.ObjV = yawObjV
        return yawObjV


def run_ZDT1():
    """================================实例化问题对象============================="""
    problem = ZDT1()            # 生成问题对象
    """==================================种群设置================================"""
    Encoding = 'RI'              # 编码方式
    NIND = 50                    # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置==============================="""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 50     # 最大进化代数
    myAlgorithm.logTras = 5      # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = False  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1      # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化=========================
    调用run执行算法模板, 得到帕累托最优解集NDSet以及最后一代种群。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值; NDSet.Phen为对应的决策变量值。
    详见Population.py中关于种群类的定义。
    """
    [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
    NDSet.save()  # 把非支配种群的信息保存到文件中


if __name__ == "__main__":

    config = {"theta": 270.,
              "inflow": 10.,
              "turb": 0.077, }

    # turbine position: [[x1, y1], [x2, y2], ...]
    layout = layout_generator()
    print(layout)

    # # yaw variables: [[yaw1, ind1], [yaw2, ind2], ...]
    # powers = ZDT1(config, layout).aimFunc()
    # print('Pop Power:', 1 / powers[:, 0], '\n')
    # print(f'Pop ObjVal Matrix: Shape = {powers.shape}\n', powers, '\n')

    run_ZDT1()

    # config = {
    #     "num": 5,
    #     "param": "horns",
    #     "inflow": 8.0,
    #     "theta": 0.,
    #     "sector": 3,
    #     "Iam": 0.077,
    #     "winds": "horns",
    #     "velocity": "Jensen",
    #     "combination": "SS",
    #     "turbulence": None,
    #     "superposition": None,}

    # layouts = np.array([[800, 3040],
    #                [800, 2480],
    #                [800, 1920],
    #                [800, 1360],
    #                [800, 800]])

    # inputs = np.array([[-10, 0.276],
    #                 [10, 0.276],
    #                 [10, 0.152],
    #                 [10, 0.276],
    #                 [10, 0.114]])

    # powers = YawedLayoutPower(config).yawed_data_generation(layouts, inputs)
    # print(powers)

    # powers = upow.LayoutPower(config).yawed_data("output/21_4_23")
    # print(powers)

