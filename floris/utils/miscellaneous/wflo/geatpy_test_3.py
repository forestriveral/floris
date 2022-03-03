import numpy as np
import geatpy as ea


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'BNH'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins
        Dim = 2  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [5, 3]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界
        ubin = [1] * Dim  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb,
                            ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        x1 = Vars[:, [0]]  # 注意这样得到的x1是一个列向量，表示所有个体的x1
        x2 = Vars[:, [1]]
        f1 = 4*x1**2 + 4*x2**2
        f2 = (x1 - 5)**2 + (x2 - 5)**2
        # 采用可行性法则处理约束
        pop.CV = np.hstack([(x1 - 5)**2 + x2**2 - 25,
                            -(x1 - 8)**2 - (x2 - 3)**2 + 7.7])
        # 把求得的目标函数值赋值给种群pop的ObjV
        pop.ObjV = np.hstack([f1, f2])

    def calReferObjV(self):  # 计算全局最优解
        N = 10000  # 欲得到10000个真实前沿点
        x1 = np.linspace(0, 5, N)
        x2 = x1.copy()
        x2[x1 >= 3] = 3
        return np.vstack((4 * x1**2 + 4 * x2**2,
                          (x1 - 5)**2 + (x2 - 5)**2)).T


"""=======================实例化问题对象==========================="""
problem = MyProblem()  # 实例化问题对象
"""=========================种群设置=============================="""
Encoding = 'RI'  # 编码方式
NIND = 100  # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,
                  problem.borders)  # 创建区域描述器
population = ea.Population(Encoding, Field, NIND)
# 实例化种群对象（此时种群还没被真正初始化，仅仅是生成一个种群对象）
"""=========================算法参数设置============================"""
myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
myAlgorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
myAlgorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
myAlgorithm.MAXGEN = 200  # 最大进化代数
myAlgorithm.logTras = 10  # 设置每多少代记录日志，若设置成0则表示不记录日志
myAlgorithm.verbose = False  # 设置是否打印输出日志信息
myAlgorithm.drawing = 1
# 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画； 3：绘制决策空间过程动画）
"""==========================调用算法模板进行种群进化==============
调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。
NDSet是一个种群类Population的对象。
NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
详见Population.py中关于种群类的定义。
"""
[NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
# NDSet.save()  # 把非支配种群的信息保存到文件中
"""===========================输出结果========================"""
print('用时：%s 秒' % myAlgorithm.passTime)
print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
if myAlgorithm.log is not None and NDSet.sizes != 0:
    print('GD', myAlgorithm.log['gd'][-1])
    print('IGD', myAlgorithm.log['igd'][-1])
    print('HV', myAlgorithm.log['hv'][-1])
    print('Spacing', myAlgorithm.log['spacing'][-1])
"""======================进化过程指标追踪分析=================="""
metricName = [['igd'], ['hv']]
Metrics = np.array([myAlgorithm.log[metricName[i][0]] for i in
                    range(len(metricName))]).T
# 绘制指标追踪分析图
ea.trcplot(Metrics, labels=metricName, titles=metricName)
