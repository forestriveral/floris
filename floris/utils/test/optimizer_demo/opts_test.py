import os, sys
import numpy as np
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt

from sko.DE import DE
from sko.GA import GA
from sko.PSO import PSO
from sko.SA import SA
from sko.ACA import ACA_TSP
from sko.IA import IA_TSP
from sko.AFSA import AFSA


num_points = 50

points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
# print(points_coordinate)
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
# print(distance_matrix)


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


def de_test():
    '''
    min f(x1, x2, x3) = x1^2 + x2^2 + x3^2
    s.t.
        x1*x2 >= 1
        x1*x2 <= 5
        x2 + x3 = 1
        0 <= x1, x2, x3 <= 5
    '''

    def obj_func(p):
        x1, x2, x3 = p
        return x1 ** 2 + x2 ** 2 + x3 ** 2

    constraint_eq = [
        lambda x: 1 - x[1] - x[2]
    ]

    constraint_ueq = [
        lambda x: 1 - x[0] * x[1],
        lambda x: x[0] * x[1] - 5
    ]
    
    de = DE(func=obj_func, n_dim=3, size_pop=50,
            max_iter=800, lb=[0, 0, 0], ub=[5, 5, 5],
            constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)

    best_x, best_y = de.run()
    print('best_x:', best_x, 'best_y:', best_y)


def sa_test():
    demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
    
    sa = SA(func=demo_func, x0=[1, 1, 1], T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
    best_x, best_y = sa.run()
    print('best_x:', best_x, 'best_y', best_y)
    
    plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
    plt.show()


def aca_test():
    aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
                  size_pop=50, max_iter=200,
                  distance_matrix=distance_matrix)
    
    # print(distance_matrix)
    best_x, best_y = aca.run()
    print('best_x:', best_x, '\nbest_y:', best_y)
    

def ia_test():
    ia_tsp = IA_TSP(func=cal_total_distance, n_dim=num_points,
                    size_pop=500, max_iter=800, prob_mut=0.2,
                    T=0.7, alpha=0.95)
    best_points, best_distance = ia_tsp.run()
    print('best routine:', best_points, 'best_distance:', best_distance)


def afsa_test():
    
    def func(x):
        x1, x2 = x
        return 1 / x1 ** 2 + x1 ** 2 + 1 / x2 ** 2 + x2 ** 2

    afsa = AFSA(func, n_dim=2, size_pop=50, max_iter=300,
                max_try_num=100, step=0.5, visual=0.3,
                q=0.98, delta=0.5)
    best_x, best_y = afsa.run()
    print(best_x, best_y)



if __name__ == '__main__':
    
    sa_test()
