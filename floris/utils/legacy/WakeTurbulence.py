import numpy as np
import matplotlib.pyplot as plt

from floris.utils.module.tools import plot_property as ppt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def Quart(C_t, I_0, x):
    x_N = 1.
    I_add = 4.8 * (C_t**0.7) * (I_0**0.68) * ((x/x_N)**-0.57)
    I_w = np.sqrt(I_add**2 + I_0**2)
    return I_add, I_w


def Crespo(C_t, I_0, x_D):  #  (C_t, I_0, x_D, u_0, )
    # 0.1 < a <0.4, 0.07 < I_0 < 0.14, 5 < x/D < 15
    a = (1 - np.sqrt(1 - C_t)) / 2
    I_add = 0.73 * (a**0.8325) * (I_0**0.0325) * ((x_D)**-0.32)
    # I_w = np.sqrt(I_add**2 + I_0**2)
    return I_add


def Frandsen_turb(C_t, I_0, x_D):
    K_n = 0.4
    # print(x_D)
    I_add = np.sqrt(K_n * C_t) / x_D
    # I_w = np.sqrt(I_add**2 + I_0**2)
    return I_add


def Larsen_turb(C_t, I_0, x_D):
    I_add = 0.29 * (x_D**(-1/3)) * np.sqrt(1 - np.sqrt(1 - C_t))
    # I_w = np.sqrt(I_add**2 + I_0**2)
    return I_add


def Tian(C_t, I_0, x_D):
    K_n = 0.4
    I_add = K_n * C_t / x_D
    # I_w = I_add + I_0
    return I_add


def Gao(C_t, I_0, x_D):
    K_n = 0.4
    I_w = (K_n * C_t * (x_D**-0.5) + (I_0**0.5))**2
    return I_w


def IEC(C_t, I_0, x_D):
    # I_add_1 = 0.9 / (1.5 + 0.3 * x_D * (u_0**(-0.5)))
    # I_w_1 = np.sqrt(I_add_1**2 + I_0**2)

    I_add_2 = 1. / (1.5 + 0.8 * x_D * (C_t**(-0.5)))
    # I_w_2 = np.sqrt(I_add_2**2 + I_0**2)
    return I_add_2


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                            MISCELLANEOUS                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def tim_plot():
    # fig = plt.figure(figsize=(10, 8), dpi=100)
    # ax = fig.add_subplot()
    fig, axarr = plt.subplots(1, 3, sharex=True, figsize=(15, 8), dpi=100)
    axarr = axarr.flatten()

    I_0 = 0.077
    C_ts = [0.25, 0.50, 0.75]
    x_D = np.arange(5, 15, 0.1)
    tims = [Crespo, Frandsen_turb, Larsen_turb, IEC]
    colors = ['k', 'r', 'b', 'g']
    styles = ['--', '-', '-.', ':']
    labels = ['Crespo', 'Frandsen', 'Larsen', 'IEC']

    for i, C_t in enumerate(C_ts):
        for j, tim in enumerate(tims):
            # turb = tim(C_t, I_0, x_D)
            turb = np.sqrt(tim(C_t, I_0, x_D) ** 2 + I_0 ** 2)
            axarr[i].plot(x_D, turb,
                          lw=3.0, c=colors[j],
                          linestyle=styles[j],
                          label=labels[j],)
            axarr[i].set_title(f'Thrust coefficient = {C_t:.2f}', ppt.font15)
            axarr[i].set_xlabel(r'Normalized downstream distance($\mathit{D}$)', ppt.font15)
            # axarr[i].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5,])
            axarr[i].set_yticks(np.arange(0.07, 0.2, 0.02))
            ticks = axarr[i].get_xticklabels() + axarr[i].get_yticklabels()
            [tick.set_fontname('Times New Roman') for tick in ticks]
            axarr[i].tick_params(labelsize=15, direction='in')
    axarr[0].set_ylabel('The wake turbuence intensity', ppt.font15)
    axarr[0].legend(loc='upper left', prop=ppt.font15, edgecolor='None', frameon=False,
                    labelspacing=0.4, bbox_transform=axarr[0].transAxes)
    # plt.savefig('./tim_plot.png', format='png', dpi=300, bbox_inches='tight', )
    plt.show()


if __name__ == "__main__":
    # I_add = Crespo(0.276, 0.077, 15)
    # print(I_add)
    # print(np.sqrt(I_add**2 + 0.077**2))

    # I_add, I_w = Frandsen(0.8, 0.077, 10)
    # print(I_add, I_w)
    tim_plot()
