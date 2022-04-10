import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.pyplot import MultipleLocator

import gauss_turb_params as gp


data_path = 'C:\Users\Li Hang\OneDrive - CivilEng\8_Desktop\CSSC Wake Simulation'

C_t = {4:0.8803, 6:0.7721, 10.4:0.6589, 15:0.1830, 18:0.1076}

label_font = {'family': 'Times New Roman', 'size': 18}
legend_font = {'family': 'Times New Roman', 'size': 14}
title_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 15}


def parameters(x_D, vel, turb, verbose=False, **kwargs):
    params = {}
    C = C_t[vel]
    
    params["k"] = 0.11 * C**1.07 * turb**0.20
    params["ep"] = 0.23 * C**-0.25 * turb**0.17
    params["d"] = 2.3 * C**-1.2
    params["e"] = 1.0 * turb**0.1
    params["f"] = 0.7 * C**-3.2 * turb**-0.45
    
    params["o_D"] = params["k"] * x_D + params["ep"]
    params["B"] = 1. / (params["d"] + params["e"] * x_D + params["f"] * (1 + x_D)**-2)
    
    if verbose:
        print("=============")
        print("k: ", params["k"])
        print("ep: ", params["ep"])
        print("d: ", params["d"])
        print("e: ", params["e"])
        print("f: ", params["f"])
        print("\no_D: ", params["o_D"])
        print("B: ", params["B"])
        print("=============\n")
    
    return params


def Gaussian(x_D, vel, turb, D=0.608, verbose=False, mod=None):
    params = parameters(x_D, vel, turb / 100, verbose=False)
    r_D = (np.concatenate((np.arange(0, 2.4, 0.0024), np.array([2.4]))) - 1.2) / D

    if mod:

        k11, k22 = 0.8, 0.8

        def kk_1(r_D):
            return np.where(r_D <= 0.5, np.cos(np.pi / 2 * (r_D - 0.5)) ** 2, 1.)

        def kk_2(r_D):
            return np.where(r_D <= 0.5, np.cos(np.pi / 2 * (r_D + 0.5)) ** 2, 0.)

        def d_func(r_D):
            if vel == 6 and turb == 6:
                # return -0.002184 * r_D**5 + 0.006151 * r_D**4 + 0.01097 * r_D**3 \
                #     - 0.0308 * r_D**2 - 0.01285 * r_D + 0.02498
                return (-0.0003991 * r_D**5 + 0.001132 * r_D**4 + 0.002057 * r_D**3 \
                        - 0.005545 * r_D**2 - 0.002459 * r_D + 0.004939) * mod["dd_f"]
            elif vel == 6 and turb == 2:
                return (-3.705e-05 * r_D**9 + 0.0008563 * r_D**8 + 0.0005175 * r_D**7 \
                    -0.007533 * r_D**6 - 0.002507 * r_D**5 + 0.02278 * r_D**4 \
                        + 0.005 * r_D**3 - 0.02744 * r_D**2 - 0.003441 * r_D + 0.01094) * mod["dd_f"]
                # return 0.
            elif vel == 4 and turb == 6:
                return 0.
            elif vel == 4 and turb == 2:
                return 0.
            elif vel == 10.4 and turb == 6:
                return (4.975e-05 * r_D**5 + 0.001396 * r_D**4 - 0.0002462 * r_D**3 \
                        - 0.006325 * r_D**2 + 0.0002128 * r_D + 0.005184) * mod["dd_f"]
            elif vel == 10.4 and turb == 2:
                return (-0.0001885 * r_D**5 + 0.001067 * r_D**4 + 0.0009995 * r_D**3 \
                        - 0.005126 * r_D**2 - 0.001216 * r_D + 0.005361) * mod["dd_f"]
            else:
                # print("==> No modification of d_f!")
                return 0.

        k1, k2 = kk_1(r_D), kk_2(r_D)
        # print(k1, k2)
        # d_f = np.vectorize(d_func)(r_D)
        d_f = gauss_mod(r_D, gp.gauss_params[f"{vel}_{turb}"], x_D)
        # d_f = 0
        # print(d_f)

        new_B = params["B"] * mod["d_B"]
        new_o_D = params["o_D"] * mod["d_o_D"]
        mean_offset = mod["d_m"]

        gauss_left = k1 * np.exp(- ((r_D - 1 / 2 - mean_offset)**2) / (2 * new_o_D**2))
        gauss_right = k2 * np.exp(- ((r_D + 1 / 2 + mean_offset)**2) / (2 * new_o_D**2))

        # added_I = new_B * gauss_left
        # added_I = new_B * gauss_right
        added_I = new_B * ((gauss_left + gauss_right))
        total_I = np.sqrt(added_I**2 + (turb / 100)**2 + d_f)

    else:
        def k_1(r_D):
            return np.where(r_D <= 0.5, np.cos(np.pi / 2 * (r_D - 0.5)) ** 2, 1.)

        def k_2(r_D):
            return np.where(r_D <= 0.5, np.cos(np.pi / 2 * (r_D + 0.5)) ** 2, 0.)

        k1, k2 = k_1(r_D), k_2(r_D)
        added_I = params["B"] * (k1 * np.exp(- ((r_D - 1 / 2)**2) / (2 * params["o_D"]**2)) +
                                 k2 * np.exp(- ((r_D + 1 / 2)**2) / (2 * params["o_D"]**2)))

        total_I = np.sqrt(added_I**2 + (turb / 100)**2)

    return r_D, added_I, total_I


def data_package(vel, turb, x_Ds, verbose=False):
    assert isinstance(x_Ds, dict)
    data = {}
    if x_Ds:
        for x_D in x_Ds.keys():
            # print("x_D: ", x_D)
            _, added_I, wake_I = Gaussian(x_D, vel, turb, mod=x_Ds[x_D], verbose=verbose)
            data[x_D] = [added_I, wake_I]
    return data


def Gaussian_single_plot(dist):
    fig = plt.figure(num=1, figsize=(18, 6), dpi=100)
    ax = plt.subplot2grid((1, 4), (0, 0), colspan=4)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i, d in enumerate(dist):
        r_D, added_I = Gaussian(d, )
        # print("added_I: ", added_I.shape, "\n", added_I)
        ax.plot(r_D, added_I, color=colors[i], linewidth=2.0,
                linestyle="-", label="{}D".format(d))
    plt.legend()
    plt.show()


def gauss_mod(x, params, x_D):
    return (params["d1"] + params["d2"] * x_D) * (1 / (np.sqrt(2 * np.pi) * params["a"])) * \
        (np.exp(- x**2 / params["b"]**2) + params["c"])


def draw(ax, sowfa, gauss, r_D, i, **kwargs):
    if i == 5:
        ax.plot(sowfa, r_D, c='r', linestyle='-', lw=1.5,
                label=f'SOWFA')
        ax.plot(gauss, r_D, c='k', linestyle='--', lw=1.5,
                label=f"Gaussian")
        if kwargs["plot"] is not None:
            ax.plot(kwargs["plot"], r_D, c='b', linestyle='-', lw=2,
                label=f"Modified Gaussian")
    else:
        ax.plot(sowfa, r_D, c='r', linestyle='-', lw=1.5,)
        ax.plot(gauss, r_D, c='k', linestyle='--', lw=1.5,)
        if kwargs["plot"] is not None:
            ax.plot(kwargs["plot"], r_D, c='b', linestyle='-', lw=2,)


def axes_config(ax, **kwargs):
    delta = kwargs["delta"]
    ax.xaxis.set_major_locator(MultipleLocator(delta))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    # ax.set_xlim((1.7*delta, 11*delta)), ax.set_ylim((-2.01, 2.01))
    ax.set_xlim((4.7*delta, 11*delta)), ax.set_ylim((-1.51, 1.51))
    # ax.set_xticks([2*delta, 3*delta, 4*delta, 5*delta, 6*delta, 7*delta, 8*delta, 9*delta, 10*delta])
    # ax.set_xticklabels(['2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', '10D'])
    ax.set_xticks([5*delta, 6*delta, 7*delta, 8*delta, 9*delta, 10*delta])
    ax.set_xticklabels(['5D', '6D', '7D', '8D', '9D', '10D'])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(axis='x', direction='in', width=2, length=5, labelsize=14)
    ax.tick_params(axis='y', direction='in', width=2, length=5, labelsize=14)
    ax.set_xlabel('x/D', fontdict=label_font)
    ax.set_ylabel('r/D', fontdict=label_font, labelpad=-5)
    ax.legend(prop=legend_font, loc=1, edgecolor='none')
    if kwargs.get("title", False):
        ax.set_title(kwargs["title"], fontdict=title_font)
    for p in ['right', 'top', 'bottom', 'left']:
        ax.spines[p].set_linewidth(3)


def load_dirname(vel, turb, yaw=0):
    sowfa_yaw = f"yaw{yaw}" if yaw >= 0 else f"yaw({yaw})"
    sowfa = f"{data_path}/{vel}ms/{vel}-{turb}%-{sowfa_yaw}"
    gaussian = f"{data_path}/{vel}-{turb}-{yaw}.csv"
    return sowfa, gaussian


def wake_plot(vel, turb, yaw=0, D=0.608, delta=30, type="total",
              verbose=False, psave=False, pshow=True, esave=False,
              data=True, **kwargs):
    if data:
        data = data_package(vel, turb, gp.params(vel, turb, yaw))
    sowfa_path, gaussian_path = load_dirname(vel, turb, yaw)
    r_D = (np.concatenate((np.arange(0, 2.4, 0.0024), np.array([2.4]))) - 1.2) / D
    fig = plt.figure(num=1, figsize=(18, 6), dpi=140)
    ax = fig.add_subplot(111)
    title = f"Wake validation for added turbulence with U={vel}m/s, I={turb}%, Yaw={yaw}°"
    recorder = np.zeros((6, r_D.shape[0]))
    for i in range(5, 11):
        if vel == 4:
            sowfa_fname = f'x={i}D.csv'
        else:
            sowfa_fname = f'x={i}D-z=0.38.csv'
        
        sowfa = np.loadtxt(f"{sowfa_path}/{sowfa_fname}", skiprows=1,
                           delimiter=',', usecols=7, unpack=True)
        sowfa = sowfa**0.5 / vel if type == "total" \
            else np.sqrt((sowfa**0.5 / vel)**2 - (turb / 100)**2)
        sowfa_turb = sowfa * 100 + i * delta
        
        if not os.path.exists(gaussian_path):
            gauss = Gaussian(5 + i, vel, turb)[2]
        else:
            gauss = np.loadtxt(gaussian_path, delimiter=',', usecols=i + 11,
                           unpack=True)
        gauss = gauss if type == "total" else np.sqrt(gauss**2 - (turb / 100)**2)
        gauss_turb = gauss * 100 + i * delta
        
        modified_turb = None
        if data:
            if i in data.keys():
                modified_turb = data[i][0] if type == "added" else data[i][1]
                new_turb = modified_turb * 100 + i * delta
                recorder[i - 5, :] = sowfa**2 - modified_turb**2
        
        # print(i, sowfa_fname, i + 11)
        if verbose and i == 2:  # debug option
            print("sowfa: ", sowfa.shape, "\n", sowfa)
            print("sowfa_turb: ", sowfa_turb.shape, "\n", sowfa_turb)
            
            print("gauss: ", gauss.shape, "\n", gauss)
            print("gauss_turb: ", gauss_turb.shape, "\n", gauss_turb)
            print("r_D:", r_D.shape, r_D)
        
        draw(ax, sowfa_turb, gauss_turb, r_D, i, plot=new_turb)
    axes_config(ax, delta=delta, title=title)
    
    if psave:
        plt.savefig(f"pics/{vel}-{turb}-{yaw}.png", format='png',
                    dpi=200, bbox_inches='tight')
        print(f"** Picture {vel}-{turb}-{yaw} Save Done ! **")
    
    if pshow:
        plt.show()
        
    if esave:
        # print("recorder", recorder)
        fname = f"errors/{vel}_{turb}_square_errors.npy"
        np.save(fname, recorder)
        error_plot(vel, turb)


def error_plot(vel, turb, fitting=False, **kwargs):
    errors = np.load(f"errors/{vel}_{turb}_square_errors.npy")
    r_D = (np.concatenate((np.arange(0, 2.4, 0.0024), np.array([2.4]))) - 1.2) / 0.608
    fig = plt.figure(num=1, figsize=(18, 6), dpi=100)
    ax = plt.subplot2grid((1, 3), (0, 0), colspan=3)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i in range(errors.shape[0]):
        # print("added_I: ", added_I.shape, "\n", added_I)
        ax.plot(r_D, errors[i, :], color=colors[i], linewidth=1.5,
                linestyle="-", label="{}D".format(i + 5))
    if fitting:
        params = gp.gauss_params[f"{vel}_{turb}"]
        ax.plot(r_D, gauss_mod(r_D, params), color="k", linewidth=2,
                linestyle="--", label="Gauss_mod")
        
    ax.set_xlim((-1.51, 1.51))
    plt.legend()
    plt.savefig(f"errors/{vel}-{turb}.png", format='png',
                    dpi=120, bbox_inches='tight')
    plt.show()




if __name__ == "__main__":
    vel, turb, yaw = 4, 6, 0

    # wake_plot(vel, turb, yaw, psave=True, pshow=True)

    # tp.wake_plot(vel, turb, yaw, psave=False, pshow=True, esave=True)
    # tp.error_plot(vel, turb, fitting=True)
