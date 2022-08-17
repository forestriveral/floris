import os, platform
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.colors as mcolors

platform_sys = platform.uname()[0]

if platform_sys == "Windows":
    data_path = 'C:/Users/Li Hang/OneDrive - CivilEng/8_Desktop/CSSC Wake Simulation/SOWFA_data'
elif platform_sys == "Darwin":
    data_path = '/Users/forestriveral/Documents/OneDrive - CivilEng/8_Desktop/CSSC Wake Simulation/SOWFA_data'


D_rotor = 0.608
C_t = {4:0.8803,
       6:0.7721,
       10.4:0.6589,
       15:0.1830,
       18:0.1076}
r_d = (np.concatenate((np.arange(0, 2.4, 0.0024),
                       np.array([2.4]))) - 1.2) / D_rotor

dist_font = {'family': 'Times New Roman',
             'weight': 'normal',
             'style': 'italic',
             'size': 18,
             'color': 'k', }

label_font = {'family': 'Times New Roman',
              'size': 18}

legend_font = {'family': 'Times New Roman',
               'size': 15}


def Gauss_velocity(vel, turb, x_D, r_D, *args):
    vel = 10.4 if vel == 10 else vel
    thrust = C_t[vel]
    p1, p2, p3 = args[0], args[1], args[2]
    # p1, p2, p3 = 0.11, 0.23, 0.15
    k = p1 * thrust**1.07 * turb**0.20
    ep = p2 * thrust**-0.25 * turb**0.17
    a = 4 * thrust**-0.5 * ep
    b = 4 * thrust**-0.5 * k
    c = p3 * thrust**-0.25 * turb**-0.7
    sigma_D = k * x_D + ep
    A = 1. / (a + b * x_D + c * (1 + x_D)**-2)**2

    return A * np.exp(- r_D**2 / (2 * sigma_D**2))


def Non_Gauss_velocity(vel, turb, x_D, r_D, *args):
    # p1, p2, p3 = args[0], args[1], args[2]
    thrust = C_t[vel]
    p1, p2, p3 = 0.11, 0.23, 0.15
    k = p1 * thrust**1.07 * turb**0.20
    ep = p2 * thrust**-0.25 * turb**0.17
    a = 4 * thrust**-0.5 * ep
    b = 4 * thrust**-0.5 * k
    c = p3 * thrust**-0.25 * turb**-0.7
    sigma_D = k * x_D + ep
    A = 1. / (a + b * x_D + c * (1 + x_D)**-2)**2
    # A = 1. / (a + b * x_D + c * (1 + x_D)**-2)**2

    p4, p5, p6 = args[0], args[1], args[2]

    # % Non_Gauss_deficit = Non_Gauss_velocity(vel, turb, x_D, r_D, p4, p5, p6)
    # rfa_3 = [0,-0.4,-0.7,-0.8,1,1,1,1,1];%偏度
    # rfa_4 = [2.0,2.5,2.8,3.05,4,4,4,4,4];%峰度
    # % miu = [1,1,1,1,1,1,1,1,1];%均值
    # p8 = 0.02;
    # p9 = 0.015;
    # miu = 0.19;
    #     k0_1 = p8 * thrust**1.07 * turb**0.20
    #     ep0_1 = p9 * thrust**-0.25 * turb**0.17
    #     sigma_D_1 = k0_1 * x_D + ep0_1
    # r_D_1 = (r_d - miu)/sigma_D_1
    # r_D_2 = r_d - miu;
    # numberc = 1.585;
    # Non_Gauss_deficit = zeros(length(streamdist), length(r_D_1));
    # for i = 1:1:length(streamdist)
    #     if rfa_4(i)>3
        #     h_3(i) = (rfa_3(i)/6) * ((1 - 0.015 * abs(rfa_3(i)) + 0.3 * rfa_3(i)^2) * (1 + 0.2 * (rfa_4(i) -3))^-1);
        #     h_40(i) = 0.1 * ((1 + 1.25 * (rfa_4(i) - 3)^(1/3)) - 1);
        #     h_4(i) = h_40(i) * (1 - (1.43 * rfa_3(i)^2 ) * (rfa_4(i) -3)^-1)^(1 - 0.4 * rfa_4(i)^0.8);
        #     k(i) = (1 + 2 * h_3(i)^2 + 6 * h_4(i)^2)^-0.5;
        #     a(i) = h_3(i) * (3 * h_4(i))^-1;
        #     b(i) = (3 * h_4(i))^-1;
        #     c(i) = (b(i) - 1 - a(i)^2)^3;
    # Non_Gauss_deficit(i, :)= (r_D_2.^2 + numberc) .* (2 * pi)^-0.5 .* exp(-0.5 * r_D_1.^2) .* (abs(k(i) * (1 + 2 * h_3(i) * r_D_1 + 3 * h_4(i) * r_D_1.^2 - 3 * h_4(i)))).^-1;
    #     else
    #         fai(i) = (1 - 0.06 * (3 - rfa_4(i)))^(1/3);
    # b_2(i) = fai(i) * (1 - (rfa_3(i)^4 + 1.2 * rfa_3(i)^2 - 0.18) * (7.5 * exp(0.5 * rfa_4(i)))^-1);
    # b_3(i) = - (0.8 * rfa_3(i)^5 + rfa_3(i)^3 + 0.77 * rfa_3(i)) * ((rfa_4(i) - 1)^2 + 0.5)^(-1);
    # b_4(i) = -fai(i) * (0.04 - (11.5 * rfa_3(i)^4 + 6.8 * rfa_3(i)^2 +3.5) * ((rfa_4(i)^2 + 0.4)^2 + 0.15)^-1);

    # a_1(i) = b_3(i) * (3 * b_4(i))^-1;
    # b_1(i) = (b_2(i) - b_3(i) * rfa_3(i) - b_4(i) * rfa_4(i)) * (3 * b_4(i))^(-1);
    # c_1(i) = (b_1(i) - a_1(i)^2)^3;

    # for j=1:length(r_D_1)
    # ebuseno_u(i,j) = r_D_1(j) * (2 * b_4(i))^(-1) + rfa_3(i)/2 + a_1(i) * (1.5 + 1.5 * b_1(i) - a_1(i)^2);
    # dg_u(i,j) = (1/3) * (((ebuseno_u(i,j)^2 + c_1(i))^0.5 + ebuseno_u(i,j))^(-2/3)) * (((ebuseno_u(i,j)^2) + c_1(i))^-0.5 * ebuseno_u(i,j) * (2 * b_4(i))^-1 + (2 * b_4(i))^-1) - (1/3) * (((ebuseno_u(i,j)^2 + c_1(i))^0.5 - ebuseno_u(i,j))^(-2/3)) * (((ebuseno_u(i,j)^2) + c_1(i))^-0.5 * ebuseno_u(i,j) * (2 * b_4(i))^-1 - (2 * b_4(i))^-1); 
    # Non_Gauss_deficit(i,j) = (r_D_2(j).^2 + numberc) * (2 * pi)^-0.5 * exp(-0.5 * r_D_1(j).^2) .* (abs(dg_u(i,j)))^-1;
    # end
    # end
    # end

    return A / (p4 * r_D**2 + p5 * r_D + p6) * np.exp(- r_D**2 / (2 * sigma_D**2))


def sowfa_load(inflow, distance, num=None, deficit=True):
    vel, turb, yaw = int(inflow[0]), int(inflow[1]), int(inflow[2])
    vel = 10.4 if vel == 10 else vel
    yaw = f"yaw{yaw}" if yaw >= 0 else f"yaw({yaw})"
    path = f"{data_path}/{vel}ms/{vel}-{turb}%-{yaw}"
    print('Path: ', path)
    assert isinstance(distance, list)
    vel_data = []
    for dist in distance:
        fname = f'x={dist}D.csv' if int(vel) == 4 else f'x={dist}D-z=0.38.csv'
        data = np.loadtxt(f"{path}/{fname}", skiprows=1, delimiter=',',
                          usecols=4, unpack=True)
        vel_data.append(data / vel)
    ind = np.linspace(0, len(r_d), num, endpoint=False, dtype=int) if num is not None else np.arange(len(r_d))
    if deficit:
        vel_data = [1 - vel_data[i] / np.max(vel_data[i]) for i in range(len(vel_data))]
    sowfa_sk = skew_kurtosis(vel_data)
    return [vel_data[i][ind] for i in range(len(vel_data))]


def skew_kurtosis(vel_data):
    skew = [stats.skew(vel) for vel in vel_data]
    kurtosis = [stats.kurtosis(vel) for vel in vel_data]
    return [skew, kurtosis]


def wake_plot(inflow, gauss=True, nongauss=True, ax=None):
    vel, turb, yaw = inflow
    distance = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    sowfa_data = sowfa_load(inflow, distance, num=80, deficit=True)
    # sowfa_sk = skew_kurtosis(sowfa_data)
    title = f"Wake velocity fitting with U={vel}m/s, I={turb}%, Yaw={yaw}°"
    if ax is None:
        fig, ax = plt.subplots(3, 3, sharey=True, figsize=(14, 12), dpi=80)
    for i, dist in enumerate(distance):
        axi = ax.flatten()[i]
        axi.plot(r_d[i], sowfa_data[i], c="w", lw=0., label='SOWFA',
                 markersize=6, marker="o", markeredgecolor='k', markeredgewidth=1.,)
        if nongauss:
            nongauss_vel = Non_Gauss_velocity(vel, turb, dist, r_d, 0.472, 0.07, 0.254)
            axi.plot(r_d, nongauss_vel, c='r', linestyle='-', lw=1.5, label="Non-gaussian",)
        if gauss:
            gauss_vel = Gauss_velocity(vel, turb, dist, r_d, 0.014, 0.24, 0.005)
            axi.plot(r_d, gauss_vel, c='k', linestyle='--', lw=1.5, label="Ishihara-Qian",)
        if i in [6, 7, 8]:
            axi.set_xlabel('y/D', fontdict=dist_font)
        axi.set_xlim([-1.5, 1.5])
        axi.set_xticks([-1.5, -1., -0.5, 0., 0.5, 1., 1.5])
        axi.set_xticklabels(['-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5'])
        if i in [0, 3, 6]:
            axi.set_ylabel('deficit', fontdict=dist_font, labelpad=5)
        axi.set_ylim([0., 1.])
        axi.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.])
        axi.set_yticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'])
        axi.text(0.95, 0.9, f'x/d = {dist}', va='top', ha='right',
                 fontdict=dist_font, transform=axi.transAxes, )
        labels = axi.get_xticklabels() + axi.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        axi.tick_params(labelsize=15, colors='k', direction='in',
                        top=True, bottom=True, left=True, right=True)
    ax1 = ax.flatten()[1]
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="upper left", prop=legend_font, columnspacing=0.8,
               edgecolor='None', frameon=False, labelspacing=0.5, bbox_to_anchor=(-0.40, 1.2),
               bbox_transform=ax1.transAxes, ncol=len(labels), handletextpad=0.5)
    fig.suptitle(title, position=(0.5, 0.02), va='bottom', ha='center', fontproperties=label_font)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    # plt.savefig(f"../../outputs/nongauss_test.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def parameter_fitting(model, inflow, param_range):
    p1s, p2s, p3s = param_range
    vel, turb, yaw = inflow
    distance = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    sowfa_data = sowfa_load(inflow, distance)
    # print(sowfa_data)
    error_0, solution = np.inf, []
    for p1 in np.arange(p1s[0], p1s[1], 0.001):
        for p2 in np.arange(p2s[0], p2s[1], 0.001):
            for p3 in np.arange(p3s[0], p3s[1], 0.001):
                error_1 = 0.
                for i, dist in enumerate(distance):
                    sowfa_vel = 1 - sowfa_data[i] / np.max(sowfa_data[i])
                    gauss_vel = model(vel, turb, dist, r_d, p1, p2, p3)
                    error_1 += np.sum(np.abs(gauss_vel - sowfa_vel)**2)
                if error_1 < error_0:
                    solution = [p1, p2, p3]
                    error_0 = error_1
    print('Optimal parameters: ', solution, 'Error: ', error_0,)
    return solution



if __name__ == "__main__":
    # [vel, turb, yaw]
    inflow = [4, 6, 20]
    wake_plot(inflow)
    # parameter_fitting(Non_Gauss_velocity, inflow,
    #                   [(0.42, 0.49), (0.07, 0.13), (0.22, 0.28)])
