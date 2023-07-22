# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化目标最小最大化标记列表，1：min；-1：max
        Dim = 3  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化决策变量类型，0：连续；1：离散
        lb = [0, 0, 0]  # 决策变量下界
        ub = [1, 1, 2]  # 决策变量上界
        lbin = [1, 1, 0]  # 决策变量下边界
        ubin = [1, 1, 0]  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        Vars = pop.Phen  # 得到决策变量矩阵
        x1 = Vars[:, [0]]  # 取出第一列得到所有个体的x1组成的列向量
        x2 = Vars[:, [1]]  # 取出第二列得到所有个体的x2组成的列向量
        x3 = Vars[:, [2]]  # 取出第三列得到所有个体的x3组成的列向量
        # 计算目标函数值，赋值给pop种群对象的ObjV属性
        pop.ObjV = 4*x1 + 2*x2 + x3
        # 采用可行性法则处理约束，生成种群个体违反约束程度矩阵
        pop.CV = np.hstack([2*x1 + x2 - 1,  # 第一个约束
                            x1 + 2*x3 - 2,  # 第二个约束
                            np.abs(x1 + x2 + x3 - 1)])  # 第三个约束


"""============================实例化问题对象========================"""
problem = MyProblem() # 实例化问题对象
"""==============================种群设置==========================="""
Encoding = 'RI' # 编码方式
NIND = 50 # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
population = ea.Population(Encoding, Field, NIND)
# 实例化种群对象（此时种群还没被真正初始化，仅仅是生成一个种群对象）
"""===========================算法参数设置=========================="""
myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population) # 实例化一个算法模板对象
myAlgorithm.MAXGEN = 1000 # 最大进化代数
myAlgorithm.mutOper.F = 0.5 # 差分进化中的参数F
myAlgorithm.recOper.XOVR = 0.7 # 设置交叉概率
myAlgorithm.logTras = 1 #设置每隔多少代记录日志，若设置成0则表示不记录日志
myAlgorithm.verbose = True # 设置是否打印输出日志信息
myAlgorithm.drawing = 3
# 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画； 3：绘制决策空间过程动画）
"""==========================调用算法模板进行种群进化==============="""
[BestIndi, population] = myAlgorithm.run() # 执行算法模板，得到最优个体以及最后一代种群
BestIndi.save() # 把最优个体的信息保存到文件中
"""=================================输出结果======================="""
print('评价次数：%s' % myAlgorithm.evalsNum)
print('时间已过%s 秒' % myAlgorithm.passTime)
if BestIndi.sizes != 0:
    print('最优的目标函数值为：%s' % BestIndi.ObjV[0][0])
    print('最优的控制变量值为：')
    for i in range(BestIndi.Phen.shape[1]):
        print(BestIndi.Phen[0, i])
else:
    print('没找到可行解。')