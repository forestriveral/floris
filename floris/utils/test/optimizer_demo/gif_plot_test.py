import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


myfont = fm.FontProperties(fname="/usr/share/fonts/myfonts/TIMES.TTF", size=14)


def simple_plot():
    """
    simple plot
    """
    # 生成画布
    plt.figure(figsize=(8, 6), dpi=80)

    # 打开交互模式
    plt.ion()

    # 循环
    for index in range(100):
        # 清除原有图像
        plt.cla()

        # 设定标题等
        plt.title("Dynamic Curve Plotting", fontproperties=myfont)
        plt.grid(True)

        # 生成测试数据
        x = np.linspace(-np.pi + 0.1*index, np.pi+0.1*index, 256, endpoint=True)
        y_cos, y_sin = np.cos(x), np.sin(x)

        # 设置X轴
        plt.xlabel("X axis", fontproperties=myfont)
        plt.xlim(-4 + 0.1*index, 4 + 0.1*index)
        plt.xticks(np.linspace(-4 + 0.1*index, 4+0.1*index, 9, endpoint=True))

        # 设置Y轴
        plt.ylabel("Y axis", fontproperties=myfont)
        plt.ylim(-1.0, 1.0)
        plt.yticks(np.linspace(-1, 1, 9, endpoint=True))

        # 画两条曲线
        plt.plot(x, y_cos, "b--", linewidth=2.0, label="cos demo")
        plt.plot(x, y_sin, "g-", linewidth=2.0, label="sin demo")

        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper left", prop=myfont, shadow=True)

        # 暂停
        plt.pause(0.1)

    # 关闭交互模式
    plt.ioff()

    # 图形显示
    plt.show()
    return None


def animation_plot():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots()
    line, = ax.plot(x, y, color='k')

    def update(num, x, y, line):
        line.set_data(x[:num], y[:num])
        line.axes.axis([0, 10, -1, 1])
        return line,

    ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line],
                                  interval=25, blit=False)
    ani.save('test.gif')
    plt.show()




if __name__ == '__main__':
    
    # simple_plot()
    animation_plot()