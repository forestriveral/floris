import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def poly_data_fit():
    data = np.loadtxt("../params/wflo/thrust.txt")
    x, y = data[:, 0], data[:, 1]
    # x, y = data[5:12, 0], data[5:12, 1]
    # x, y = np.array([0.5, 0.05, 0.005, 0.00005]), np.array([0.055, 0.040, 0.030, 0.031])

    f1 = np.polyfit(x, y, 12)
    print('f1 is :\n', f1)
    p1 = np.poly1d(f1)
    print('p1 is :\n', p1)

    # test_x = np.arange(4, 25, 0.5)
    yvals = p1(x)  # 拟合y值
    # print(p1(np.arange(4.1, 20, 1)))
    # print('yvals is :\n', yvals)
    # plot1 = plt.plot(x, y, 's', label='original values')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, yvals, 'r', label='polyfit values')
    plt.plot(x, y, 'g', label='original values')
    
    plt.xlabel('x', )
    plt.ylabel('y', )
    # plt.xlim((7, 10))
    plt.legend(loc=2)
    plt.title('polyfitting')
    plt.show()


def random_data_fit():
    data = np.loadtxt("../params/wflo/ct_curve.txt")
    x, y = data[:, 0], data[:, 1]

    def func(x, a, b, c):
        return a*(np.exp(-(x - b) ** 2 / (2 * c ** 2))/(math.sqrt(2*math.pi)*c))*(431+(4750/x))

    popt, pcov = curve_fit(func, x, y)
    a, b, c = popt[0], popt[1], popt[2]

    yvals = func(x, a, b, c)
    print(u'a:', a)
    print(u'b:', b)
    print(u'c:', c)

    # 绘图
    plot1 = plt.plot(x, y, 's', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=3)  # 指定legend的位置右下角
    plt.title('curve_fit')
    plt.show()



if __name__ == "__main__":
    
    poly_data_fit()
