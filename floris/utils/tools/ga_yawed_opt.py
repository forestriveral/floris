import multiprocessing as mp
from multiprocessing import Pool as ProcessPool

import geatpy as ea
import numpy as np
import pandas as pd

from floris.utils.modules.wflo_layout import LayoutPower
from floris.utils.tools import valid_ops as vops
from floris.utils.tools import farm_config as fconfig

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class YawedLayoutPower(LayoutPower):
    def __init__(self, configs, **kwargs):
        super().__init__(configs, **kwargs)

    def yawed_power(self, layouts, yaweds):
        if self.configs['num'] != layouts.shape[0]:
            self.configs['num'] = layouts.shape[0]
        self.initialization(layouts, yawed=yaweds)
        velocity = single_yawed(self.configs, self.params,
                                self.yawed, self.layout)
        power_curve = fconfig.Horns.pow_curve
        powers = np.vectorize(power_curve)(velocity)
        return powers

    def yawed_generator(self, layouts, N=500):
        yawed_data = vops.random_yawed_generator(layouts.shape[0], N)
        power_data = np.zeros((N, layouts.shape[0]))
        for i in range(N):
            power_data[i, :] = self.yawed_power(
                layouts, yawed_data[i, :, :])
        return yawed_data, power_data

    def yawed_data(self, dsave=None):
        layouts = vops.yawed_layout_generator()
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


def single_yawed(config, params, yawed, layout):
    theta_w = config['theta']
    wt_loc = vops.coordinate_transform(layout, theta_w)
    wt_index = vops.wind_turbines_sort(wt_loc)
    # print(wt_index)
    assert len(wt_index) == wt_loc.shape[0]
    deficits = np.zeros(len(wt_index))
    deficit_tab = np.full((len(wt_index), len(wt_index) + 2), None)
    turbulence_tab = np.full((len(wt_index), len(wt_index) + 2), None)
    for i, t in enumerate(wt_index):
        if i == 0:
            deficit_tab[0, -2], deficit_tab[0, -1] = 0., float(config["inflow"])
            if config["tim"] is not None:
                turbulence_tab[0, -2], turbulence_tab[0, -1] = 0., config["Iam"]
        wake_model = vops.find_and_load_model(config["wm"], model="wm")
        ct_t, ytheta = 4 * yawed[t, 1] * (1 - yawed[t, 1]), yawed[t, 0]
        wake = wake_model(wt_loc[t, :], deficit_tab[i, -1], ct_t,
                          params.iloc[t]["D_r"], params.iloc[t]["z_hub"], T_m=config["tim"],
                          I_w=turbulence_tab[i, -1], I_a=config["Iam"], ytheta=ytheta)
        if i < len(wt_index) - 1:
            for j, wt in enumerate(wt_index[i+1:]):
                deficit_tab[i, i + j + 1], turbulence_tab[i, i + j + 1] = \
                    wake.wake_loss(wt_loc[wt, :], params.iloc[wt]["D_r"], debug=None)
            multiple_wake_model = vops.find_and_load_model(config["wsm"], model="wsm")
            total_deficit = multiple_wake_model(deficit_tab[:, :], i + 1,
                                                inflow=float(config["inflow"]))
            if config["tim"] is not None:
                if np.max(turbulence_tab[:i+1, i+1]) is None:
                    raise ValueError("Invalid value in turbulence table!")
                if config["tim"] == "Modified":
                    turbulence_tab[i + 1, -2] = np.sum(turbulence_tab[:i+1, i+1])
                else:
                    turbulence_tab[i + 1, -2] = np.max(turbulence_tab[:i+1, i+1])
                turbulence_tab[i + 1, -1] = np.sqrt(
                    np.max(turbulence_tab[:i+1, i+1])**2 + config["Iam"]**2)
            deficit_tab[i + 1, -2] = total_deficit
            deficit_tab[i + 1, -1] = float(config["inflow"]) * (1 - total_deficit)
        else:
            break
        deficits[:] = vops.wt_power_reorder(wt_index, deficit_tab[:, -1])
    # print(deficit_tab)
    return deficits


class YawedPowerProblem(ea.Problem):
    def __init__(self, configs, layout):
        name = 'YawedPower'  # 初始化name（函数名称，可以随意设置）
        self.configs = configs
        self.layout = layout
        self.Dim = 2 * self.configs["num"]  # 初始化Dim（决策变量维数）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1] * M  # 初始化目标最小最大化标记列表，1：min；-1：max
        varTypes = [0] * self.Dim # 初始化决策变量类型，0：连续；1：离散

        lb = [-30, 0.031] * self.configs["num"]
        ub = [30, 0.276] * self.configs["num"]
        lbin = [0] * self.Dim  # 决策变量下边界
        ubin = [0] * self.Dim  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, self.Dim,
                            varTypes, lb, ub, lbin, ubin)
        self.calculator = YawedLayoutPower(self.configs)
        self.pool = ProcessPool(int(mp.cpu_count()) - 2)  # 设置池的大小

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
        # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画； 3：绘制决策空间过程动画）
        """==========================调用算法模板进行种群进化==============="""
        [BestIndi, self.population] = myAlgorithm.run() # 执行算法模板，得到最优个体以及最后一代种群
        BestIndi.save("solution/Results") # 把最优个体的信息保存到文件中
        """=================================输出结果======================="""
        print('Evaluation times：{}'.format(myAlgorithm.evalsNum))
        print('Elapsed time %s %s' % vops.time_formator(myAlgorithm.passTime))
        if BestIndi.sizes != 0:
            yawed_data = pd.read_csv(
                f"../solution/Results/Phen.csv",
                header=None).values.reshape(self.config["num"], 2)
            print('Optimal Yawed：(Turbine: Yaw/Induction)')
            for i in range(yawed_data.shape[0]):
                print(f'   {i+1}: {yawed_data[i, 0]:.1f} / {yawed_data[i, 1]:.3f}')
            print('Optimal Power：%s' % (BestIndi.ObjV[0][0]))
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
        "wm": "Jensen",
        "wsm": "SS",
        "tim": None,
        "tsm": None,}
    
    layout_5, layout_9, layout_25 = vops.yawed_layout_generator()
    
    YawedOpt(config, eval(f"layout_{N}")).solution()



if __name__ == "__main__":

    config = {
        "num": 5,
        "param": "horns",
        "inflow": 8.0,
        "theta": 0.,
        "sector": 3,
        "Iam": 0.077,
        "winds": "horns",
        "wm": "Jensen",
        "wsm": "SS",
        "tim": None,
        "tsm": None,}
    
    layouts = np.array([[800, 3040],
                   [800, 2480],
                   [800, 1920],
                   [800, 1360],
                   [800, 800]])

    # inputs = np.array([[-10, 0.276],
    #                 [10, 0.276],
    #                 [10, 0.152],
    #                 [10, 0.276],
    #                 [10, 0.114]])
    
    # powers = YawedLayoutPower(config).yawed_data_generation(layouts, inputs)
    # print(powers)
    
    # powers = upow.LayoutPower(config).yawed_data("output/21_4_23")
    # print(powers)