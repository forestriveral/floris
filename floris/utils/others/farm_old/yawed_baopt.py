import os
import sys
root = os.path.dirname(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.append(root)
import numpy as np
import matplotlib.pyplot as plt
import import_string

from bayes_opt import BayesianOptimization



def BOA(num, iters, **kwargs):
    power_func = import_string(f"yawed_power.yawed_power_{num}")
    a = [(-30, 30), (0.031, 0.276)]
    region = {f'x{i}': a[(i - 1) % 2] for i in range(1, 2 * num + 1)}
    
    bopt = BayesianOptimization(power_func, region)
    
    # 贝叶斯优化过程：
    bopt.maximize(init_points=5,
                  n_iter=iters,
                  acq='ucb',
                  kappa=2.576,
                  kappa_decay=1,
                  kappa_decay_delay=0,
                  xi=0.0,
                  **kwargs)

    # 输出最优结果的变量组合：
    index = []
    for i in bopt.res:
        index.append(i['target'])
    max_index = index.index(max(index))

    print('第%d次迭代得到最优结果：' % (max_index+1))
    print(bopt.res[max_index])
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot()
    ax.plot(np.arange(len(index)), np.array(index), 'b-')
    plt.savefig("pictures/yawed_power_curve.png", format='png',
                dpi=300, bbox_inches='tight')


if __name__ == "__main__":

    BOA(5, 300)
