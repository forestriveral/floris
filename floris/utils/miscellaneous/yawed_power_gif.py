
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import moviepy.editor as meditor
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator

import floris.utils.visual.property as ppt


def yawed_turbine_power():
    turbine_power = np.load('./turbine_power.npy')
    wt_num = turbine_power.shape[1]
    sum_power = turbine_power.sum(axis=1).reshape(2, 1, turbine_power.shape[2])
    turbine_power = np.concatenate((turbine_power, sum_power), axis=1)

    time_point = 20
    cpow, bpow = turbine_power[0, :, time_point], turbine_power[1, :, time_point]
    normal_power = np.array([2., 2., 2., 2., 2., 2. * wt_num])
    cpow, bpow = cpow / normal_power, bpow / normal_power

    n = wt_num + 1
    bar_width = 0.11
    x = np.array([1. + i * 3. * bar_width for i in range(n)])
    # patterns = ('', 'x', '//')
    # colors = ['b', 'g', 'r']
    labels = [r"$\mathit{T_1}$",
              r"$\mathit{T_2}$",
              r"$\mathit{T_3}$",
              r"$\mathit{T_4}$",
              r"$\mathit{T_5}$",
              r"$\mathit{Total}$",]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=120, facecolor="white")
    for i in range(n):
        cpow_color = 'r' if cpow[i] - bpow[i] > 0. else 'b'
        ax.bar(x[i] + bar_width,
               cpow[i],
               bar_width,
               color=cpow_color,
               align='center',
               label='Controlled',
               #    hatch=h1,
               linewidth=1.0,
               edgecolor='k',
               alpha=1.,
               )
        ax.bar(x[i] + 2 * bar_width,
               bpow[i],
               bar_width,
               #    bottom=v1,
               color='g',
               align='center',
               label='Baseline',
               #    hatch=h2,
               linewidth=1.0,
               edgecolor='k',
               alpha=1.,
               )

    ticks = ax.get_xticklabels() + ax.get_yticklabels()
    [tick.set_fontname('Times New Roman') for tick in ticks]
    ax.tick_params(labelsize=20, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    ax.set_xlabel(r'Turbine and wind farm', ppt.font20)
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(labels)
    ax.set_ylim([0.2, 1.1])
    ax.set_ylabel(r'Normalized power (MW)', ppt.font20)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    # ax.set_yticks([x + 1.5 * bar_width])
    # ax.set_yticklabels(labels)
    # ax.text(2.2, text_yaxis[j], f"${numbers[j]}$", fontsize=15, color='k')
    # ax.legend(loc='upper right', prop=ppt.font15, edgecolor='None',
    #           frameon=False, labelspacing=0.4, columnspacing=0.6,
    #           bbox_transform=ax.transAxes, ncol=3, handletextpad=0.15)

    plt.show()


class UpdateDist:
    def __init__(self, ax, powers):
        self.wt_num = powers.shape[1]
        sum_power = powers.sum(axis=1).reshape(2, 1, powers.shape[2])
        turbine_power = np.concatenate((powers, sum_power), axis=1)

        self.cpows, self.bpows = turbine_power[0], turbine_power[1]
        self.n = self.wt_num + 1
        self.bar_width = 0.11
        self.x = np.array([1. + i * 3. * self.bar_width for i in range(self.n)])
        self.ax = ax
        self.labels = [r"$\mathit{T_1}$",
                       r"$\mathit{T_2}$",
                       r"$\mathit{T_3}$",
                       r"$\mathit{T_4}$",
                       r"$\mathit{T_5}$",
                       r"$\mathit{Total}$"]

    def __call__(self, t):
        self.ax.clear()
        cpow, bpow = self.cpows[:, t], self.bpows[:, t]
        normal_power = np.array([2., 2., 2., 2., 2., 2. * self.wt_num])
        cpow, bpow = cpow / normal_power, bpow / normal_power
        for i in range(self.n):
            cpow_color = 'r' if cpow[i] - bpow[i] > 0. else 'b'
            self.ax.bar(self.x[i] + self.bar_width, cpow[i],
                self.bar_width, color=cpow_color,
                align='center', label='Controlled',
                linewidth=1.0, edgecolor='k', alpha=1.)
            self.ax.bar(self.x[i] + 2 * self.bar_width, bpow[i],
                self.bar_width, color='g', align='center',
                label='Baseline', linewidth=1.0, edgecolor='k',
                alpha=1.)

        # Set up plot parameters
        ticks = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [tick.set_fontname('Times New Roman') for tick in ticks]
        self.ax.tick_params(labelsize=20, colors='k', direction='in',
                    top=True, bottom=True, left=True, right=True)
        self.ax.set_xlabel(r'Turbine and wind farm', ppt.font20)
        self.ax.set_xticks(self.x + 1.5 * self.bar_width)
        self.ax.set_xticklabels(self.labels)
        self.ax.set_ylim([0., 1.15])
        self.ax.set_ylabel(r'Normalized power (MW)', ppt.font20)
        self.ax.yaxis.set_major_locator(MultipleLocator(0.2))
        red_patch = mpatches.Patch(color='r', label='Controlled (+)')
        blue_patch = mpatches.Patch(color='b', label='Controlled (-)')
        green_patch = mpatches.Patch(color='g', label='Baseline')
        self.ax.legend(handles=[red_patch, blue_patch, green_patch],
                       loc='upper left', prop=ppt.font25, edgecolor='None',
                       frameon=False, labelspacing=0.8, columnspacing=1.,
                       bbox_transform=self.ax.transAxes, ncol=3,
                       handletextpad=0.5,)

        return None


def power_interpolate(powers, ratio=2.5):
    wt_num = powers.shape[1]
    powers = powers.reshape(powers.shape[0] * wt_num, powers.shape[2])
    new_powers = np.zeros((2 * wt_num, int(ratio * powers.shape[1])))
    for i in range(powers.shape[0]):
        smooth_func = interpolate.interp1d(np.arange(powers.shape[1]),
                                           powers[i], kind='quadratic')
        new_powers[i, :] = smooth_func(np.linspace(0, powers.shape[1] - 1,
                                                   new_powers.shape[1]))
    return new_powers.reshape(2, wt_num, new_powers.shape[1])


def yawed_power_gif():
    turbine_power = power_interpolate(np.load('./turbine_power.npy'))
    fig, ax = plt.subplots(figsize=(12, 8), dpi=120, facecolor="white")
    ud = UpdateDist(ax, turbine_power)
    # gif_frames = turbine_power.shape[2]
    gif_frames = 50
    ani = FuncAnimation(fig, ud, frames=gif_frames,
                        interval=500, repeat=False)
    ani.save("../outputs/turbine_power.gif", writer='imagemagick', fps=3, dpi=90)
    # plt.show()

def movie_to_gif():
    movie_file = '../outputs/24c81e86fe34038294efb222f802437f.mp4'
    clip = (meditor.VideoFileClip(movie_file).subclip(t_start=33, t_end=45).resize((640, 368)))
    clip.write_gif("../outputs/turbine_yawed.gif", fps=15)



if __name__ == "__main__":
    # yawed_turbine_power()
    # yawed_power_gif()
    movie_to_gif()
