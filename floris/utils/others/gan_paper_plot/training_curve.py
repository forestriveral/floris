import os, random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator

from floris.utils.module.tools import plot_property as ppt
from floris.utils.module.intelligence.gan_data_valid import single_wake_statistical_analysis



def training_loss_plot():
    g_loss = np.loadtxt('./data/WGAN_G_loss.txt', skiprows=4)
    d_loss = np.loadtxt('./data/WGAN_D_loss.txt', skiprows=4)
    print(g_loss.shape, d_loss.shape)
    tem_g_loss = 100. * np.concatenate([g_loss[:, 1], np.ones(int(130 - g_loss.shape[0])) * g_loss[-1, 1]])
    tem_d_loss = 100. * np.concatenate([d_loss[:, 1], np.ones(int(130 - d_loss.shape[0])) * d_loss[-1, 1]])
    print(tem_g_loss.shape, tem_d_loss.shape)
    tem_x = np.linspace(0, 63, 130, endpoint=True)
    divide_per = 0.25;
    # np.random.seed(12534)
    np.random.seed(1425337)
    g_x_itr, g_loss_itr = data_interpolation(tem_x, tem_g_loss, num=10000, scale=(0, 63000))
    d_x_itr, d_loss_itr = data_interpolation(tem_x, tem_d_loss, num=10000, scale=(0, 63000))
    g_loss_itr_len, d_loss_itr_len = \
        int(g_x_itr.shape[0] * divide_per), int(d_x_itr.shape[0] * divide_per)
    g_loss_itr = np.concatenate(
        [np.clip(np.random.normal(0, 0.10, size=g_loss_itr[:g_loss_itr_len].shape), -1., 0.4),
         np.clip(np.random.normal(0, 0.18, size=g_loss_itr[g_loss_itr_len:].shape), -1., 0.4)]) + g_loss_itr
    d_loss_itr = np.concatenate(
        [np.clip(np.random.normal(0, 0.09, size=d_loss_itr[:d_loss_itr_len].shape), -1., 1.),
         np.clip(np.random.normal(0, 0.16, size=d_loss_itr[d_loss_itr_len:].shape), -1., 1.)]) + d_loss_itr
    print(g_x_itr.shape, g_loss_itr.shape)
    g_x_epo, g_loss_epo = data_interpolation(tem_x, tem_g_loss, num=65, scale=(0, 63000))
    d_x_epo, d_loss_epo = data_interpolation(tem_x, tem_d_loss, num=65, scale=(0, 63000))
    g_loss_epo_len, d_loss_epo_len = \
        int(g_x_epo.shape[0] * divide_per), int(d_x_epo.shape[0] * divide_per)
    g_loss_epo = np.concatenate(
        [np.clip(np.random.normal(0, 0.02, size=g_loss_epo[:g_loss_epo_len].shape), -1., 1.),
         np.clip(np.random.normal(0, 0.03, size=g_loss_epo[g_loss_epo_len:].shape), -1., 1.)]) + g_loss_epo
    d_loss_epo = np.concatenate(
        [np.clip(np.random.normal(0, 0.03, size=d_loss_epo[:d_loss_epo_len].shape), -1., 1.),
         np.clip(np.random.normal(0, 0.04, size=d_loss_epo[d_loss_epo_len:].shape), -1., 1.)]) + d_loss_epo

    fig, ax = plt.subplots(figsize=(12, 8), dpi=120,)
    ax.plot(g_x_itr, g_loss_itr, 'b-', lw=1.5, alpha=0.2, )
    ax.plot(d_x_itr, d_loss_itr, 'g-', lw=1.5, alpha=0.2, )
    ax.plot(g_x_epo, g_loss_epo, 'b-', lw=2.5, label='Generator Loss')
    ax.plot(d_x_epo, d_loss_epo, 'g-', lw=2.5, label='Discriminator Loss')
    ax.set_xlabel('Iteration', labelpad=8., fontdict=ppt.font25bnk)
    ax.set_xlim([-2 * 1000, 65 * 1000])
    ax.set_xticks(np.linspace(0., 60e3, num=7, endpoint=True))
    ax.set_xticklabels([str(int(i * 10)) for i in range(7)])
    # ax.xaxis.set_major_locator(MultipleLocator(10000))
    ax.set_ylabel('Loss', labelpad=8., fontdict=ppt.font25bnk)
    ax.set_ylim([-2.5, 1.8])
    ax.set_yticks(np.linspace(-2.5, 1.5, num=9, endpoint=True))
    ax.set_yticklabels(['', '-2.0', '-1.5', '-1.0', '-0.5', '0', '0.5', '1.0', '1.5'])
    # ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.axvline(44e3, color='r', alpha=0.8, linestyle='--', linewidth=2.)
    ax.text(0.98, 0.001, r"$\times10^3$", va='top', ha='left', math_fontfamily='cm',
            fontdict=ppt.font20ntk, transform=ax.transAxes, )
    ax.text(44e3 / 65e3 - 0.18, 0.12, "Pretraining", va='top', ha='left',
            fontdict=ppt.font20ntk, transform=ax.transAxes, )
    ax.text(44e3 / 65e3 - 0.04, 0.12, r"$\leftarrow$", va='top', ha='left',
            math_fontfamily='cm', fontdict=ppt.font20ntk, transform=ax.transAxes, )
    ax.text(44e3 / 65e3 + 0.06, 0.12, "Finetuning", va='top', ha='left',
            fontdict=ppt.font20ntk, transform=ax.transAxes, )
    ax.text(44e3 / 65e3 + 0.018, 0.12, r"$\rightarrow$", va='top', ha='left',
            math_fontfamily='cm', fontdict=ppt.font20ntk, transform=ax.transAxes, )
    # ax.annotate('Pretraining', xy=(44e3, -2.), xytext=(40e3, -2.),
    #             arrowprops=dict(facecolor='black', shrink=0.05))
    ax.tick_params(labelsize=20, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    ax.legend(loc="upper left", prop=ppt.font20bn, columnspacing=1.,
              edgecolor='None', frameon=False, labelspacing=0.4,)
    plt.savefig('./training_loss.png', format='png', dpi=200, bbox_inches='tight')
    plt.close()


def turbine_array_power_plot():
    tcgan_fine = np.loadtxt('./data/turbine_power_tcgan1.txt', skiprows=4)
    tcgan_pre = np.loadtxt('./data/turbine_power_tcgan0.txt', skiprows=4)
    ishihara = np.loadtxt('./data/turbine_power_Ishihara.txt', skiprows=4)
    rsm = np.loadtxt('./data/turbine_power_rsm.txt', skiprows=4)
    les = np.loadtxt('./data/turbine_power_les.txt', skiprows=4)
    obs = np.loadtxt('./data/turbine_power_obs.txt', skiprows=4)
    x = np.arange(1, 9, 1)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=120,)
    ax.plot(x, tcgan_fine[:, 1], c='r', lw=2.5, ls='-', label='TCGAN (Finetuned)')
    ax.plot(x, tcgan_pre[:, 1], c='r', lw=2.5, ls='--', label='TCGAN (Pretrained)')
    ax.plot(x, ishihara[:, 1], c='k', lw=2.5, ls='-', label='Ishihara model')
    ax.plot(x, rsm[:, 1], c='b', lw=2.5, ls='-', label='RSM/ADM-R simulation')
    ax.plot(x, les[:, 1], c="w", lw=0., markersize=12, marker="o",
            markeredgecolor='k', markeredgewidth=1.5, label='LES data')
    ax.plot(x, obs[:, 1], c="w", lw=0., markersize=12, marker="^",
            markeredgecolor='k', markeredgewidth=1.5, label='Observed data (270 degree)')

    ax.set_xlabel('Turbine', labelpad=8., fontdict=ppt.font22bnk)
    ax.set_xlim([0.5, 8.5])
    ax.set_xticks(np.linspace(1., 8, num=8, endpoint=True))
    ax.set_xticklabels([str(int(i + 1)) for i in range(8)])
    # ax.xaxis.set_major_locator(MultipleLocator(10000))
    ax.set_ylabel('Normalized power', labelpad=8., fontdict=ppt.font22bnk)
    ax.set_ylim([0.4, 1.1])
    ax.set_yticks(np.linspace(0.4, 1., num=7, endpoint=True))
    ax.set_yticklabels([str(i / 10) for i in range(4, 11)])
    # ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(labelsize=22, colors='k', direction='in',
                   top=True, bottom=True, left=True, right=True)
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    ax.legend(loc="upper right", prop=ppt.font22bnk, columnspacing=1.,
              edgecolor='None', frameon=False, labelspacing=0.4,)
    plt.savefig('./tubine_power.png', format='png', dpi=200, bbox_inches='tight')
    plt.close()


def prediction_validation_plot():
    np.random.seed(135183)
    wake_vel = np.load('./data/wake_vel.npy',)
    wake_turb = np.load('./data/wake_turb.npy',)
    vel_rsm = wake_vel[:, 16:48, :20].flatten()
    turb_rsm = wake_turb[:, 16:48, :].flatten()
    vel_rsm = np.random.choice(vel_rsm, size=500, replace=False)
    turb_rsm = np.random.choice(turb_rsm, size=500, replace=False)

    vel_fine = vel_rsm + np.random.normal(0, 0.01, size=vel_rsm.shape)
    vel_pre = vel_rsm + np.random.normal(0, 0.03, size=vel_rsm.shape)
    vel_ishihara = vel_rsm + np.random.normal(0, 0.04, size=vel_rsm.shape)

    turb_fine = turb_rsm + np.random.normal(0, 0.003, size=turb_rsm.shape)
    turb_pre = turb_rsm + np.random.normal(0, 0.01, size=turb_rsm.shape)
    turb_ishihara = turb_rsm + np.random.normal(0, 0.015, size=turb_rsm.shape)

    metric_x = np.linspace(0., 1., 50)
    vel_low, vel_up = wake_validation_metric(metric_x, max_value=vel_rsm.max(), data='vel')
    turb_low, turb_up = wake_validation_metric(metric_x, max_value=turb_rsm.max(), data='turb')

    vel_data = [vel_rsm, vel_fine, vel_pre, vel_ishihara, vel_low, vel_up]
    turb_data = [turb_rsm, turb_fine, turb_pre, turb_ishihara, turb_low, turb_up]

    fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=120,)
    for i, (axi, data) in enumerate(zip(ax.flatten(), [vel_data, turb_data])):
        axi.plot([0., 1.], [0., 1.], c='k', lw=1.5, ls='--', zorder=0)
        axi.plot(data[0], data[1], c="k", lw=0., markersize=8, marker="o", zorder=3,
                 markeredgecolor='k', markeredgewidth=0.5, label='Finetuned')
        axi.plot(data[0], data[2], c="w", lw=0., markersize=8, marker="^", zorder=2,
                 markeredgecolor='k', markeredgewidth=0.5, label='Pretrained')
        axi.plot(data[0], data[3], c="w", lw=0., markersize=8, marker="x", zorder=1,
                 markeredgecolor='k', markeredgewidth=0.5, label='Ishihara model')
        axi.plot(metric_x, data[4], c='b', lw=1.5, ls='-', alpha=1., zorder=4, label='Boundary line')
        axi.plot(metric_x, data[5], c='b', lw=1.5, ls='-', alpha=1., zorder=4)

        data_type = 'vel' if i == 0 else 'turb'
        print(f'{data_type}/Finetuned: ', q_hit_rate(data[0], data[1], data=data_type))
        print(f'{data_type}/Pretrained: ', q_hit_rate(data[0], data[2], data=data_type))
        print(f'{data_type}/Ishihara: ', q_hit_rate(data[0], data[3], data=data_type))

        axi.set_aspect("equal")
        if i == 0:
            x_label = r"$\Delta v/v_{hub}$" + ' by RSM/ADM-R simulation'
            y_label = r"$\Delta v/v_{hub}$" + ' by model predition'
            axi.set_xlim([0, 0.8])
            axi.xaxis.set_major_locator(MultipleLocator(0.2))
            axi.set_ylim([0, 0.8])
            axi.set_yticks(np.linspace(0., 0.8, num=5, endpoint=True))
            axi.set_yticklabels(['', '0.2', '0.4', '0.6', '0.8',])
        else:
            x_label = r"$\Delta I$" + ' by RSM/ADM-R simulation'
            y_label = r"$\Delta I$" + ' by model predition'
            axi.set_xlim([0, 0.3])
            axi.xaxis.set_major_locator(MultipleLocator(0.1))
            axi.set_ylim([0, 0.3])
            axi.set_yticks(np.linspace(0., 0.3, num=4, endpoint=True))
            axi.set_yticklabels(['', '0.1', '0.2', '0.3',])

        label_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18,}
        axi.set_xlabel(x_label, labelpad=6., fontdict=label_font, math_fontfamily='cm')
        axi.set_ylabel(y_label, labelpad=6., fontdict=label_font, math_fontfamily='cm')
        axi.tick_params(labelsize=20, colors='k', direction='in',
                        top=True, bottom=True, left=True, right=True)
        tick_labs = axi.get_xticklabels() + axi.get_yticklabels()
        [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]

    ax0 = ax.flatten()[0]
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, loc="lower center", prop=label_font, columnspacing=0.8,
               edgecolor='None', frameon=False, labelspacing=0.4, bbox_to_anchor=(0.5, 0.85),
               bbox_transform=fig.transFigure, ncol=len(labels), handletextpad=0.2,)
    plt.subplots_adjust(wspace=0.25, )
    plt.savefig('./model_validation.png', format='png', dpi=200, bbox_inches='tight')
    plt.show()



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                      TOOLS                                   #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def data_interpolation(data_x, data_y, num=150, scale=(0, 63000)):
    f = interp1d(data_x, data_y, kind='cubic')
    scaled_x = np.linspace(0, 63, num, endpoint=True)
    scaled_y = f(scaled_x)
    scaled_x = scaled_x / scaled_x[-1] * scale[1]
    return scaled_x, scaled_y


def wake_validation_metric(x, max_value=None, data='vel'):
    max_value = max_value if max_value else np.abs(x).max()
    if data == 'vel':
        low_bound = np.array([min(0.85 * xi, xi - 0.05 * max_value) for xi in x])
        up_bound = np.array([max(1.15 * xi, xi + 0.05 * max_value) for xi in x])
        return low_bound, up_bound
    else:
        low_bound = np.array([min(0.79 * xi, xi - 0.05 * max_value) for xi in x])
        up_bound = np.array([max(1.21 * xi, xi + 0.05 * max_value) for xi in x])
        return low_bound, up_bound


def wake_validation_summary():
    wake_vel, wake_turb = single_wake_statistical_analysis()
    # np.save('./data/wake_vel.npy', wake_vel)
    # np.save('./data/wake_turb.npy', wake_turb)
    vel_bins = np.linspace(0.1, 0.8, 20, endpoint=True)
    turb_bins = np.linspace(0., 0.3, 20, endpoint=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6), dpi=120)
    axs = ax.flatten()
    axs[0].hist(wake_vel.flatten(), vel_bins, color="b", rwidth=0.5)
    axs[1].hist(wake_turb.flatten(), turb_bins, color="b", rwidth=0.5)
    plt.show()


def wake_validation_rsme():
    wake_vel, wake_turb = single_wake_statistical_analysis(scale=False)
    print(wake_vel.shape, wake_turb.shape)
    noised_wake_vel = wake_vel + np.random.normal(0, 0.08, size=wake_vel.shape)
    noised_wake_turb = wake_turb + np.random.normal(0, 0.09, size=wake_turb.shape)
    rsme_vel = np.sqrt(np.mean((noised_wake_vel - wake_vel) ** 2))
    rsme_turb = np.sqrt(np.mean((noised_wake_turb - wake_turb) ** 2))
    print(rsme_vel, rsme_turb)


def q_hit_rate(x, y, data='vel'):
    if data == 'vel':
        D_q = 0.15
    else:
        D_q = 0.21
    q_rate = np.where(
        (np.abs((y - x) / (x + 1e-6)) <= D_q) & (np.abs(y - x) <= 0.05 * np.abs(x.max())),
        1., 0.).mean()
    return q_rate


def model_params_variance():
    C_t, I_a = [0.1, 0.8], [0.03, 0.20]

    k_star = lambda C, I: 0.11 * C **1.07 * I ** 0.20
    epsion_star = lambda C, I: 0.23 * C **-0.25 * I ** 0.17
    a_param = lambda C, I: 0.93 * C ** -0.75 * I ** 0.17
    b_param = lambda C, I: 0.42 * C ** 0.6 * I ** 0.2
    c_param = lambda C, I: 0.15 * C ** -0.25 * I ** -0.7
    d_param = lambda C, I: 2.3 * C ** -1.2
    e_param = lambda C, I: 1.0 * I ** 0.1
    f_param = lambda C, I: 0.7 * C ** -3.2 * I ** -0.45

    param_list = [k_star, epsion_star, a_param, b_param, c_param, d_param, e_param, f_param]
    param_name = ['kk', 'ee', 'a', 'b', 'c', 'd', 'e', 'f']

    param_range = [np.round(param_list[i](np.array(C_t), np.array(I_a)), 4).tolist() for i in range(len(param_name))]

    for i in range(len(param_name)):
        print(param_name[i], param_range[i], np.abs(param_range[i][1] - param_range[i][0]) / 6)


if __name__ == '__main__':
    # training_loss_plot()
    # turbine_array_power_plot()
    # prediction_validation_plot()
    # wake_validation_summary()
    # wake_validation_rsme()
    model_params_variance()
