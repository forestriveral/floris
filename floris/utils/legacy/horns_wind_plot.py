import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import splev, splrep

from floris.utils.module.tools import plot_property as ppt


file_dir = os.path.dirname(os.path.dirname(__file__))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                       MAIN                                   #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def wind_dist_plot(wind_name="horns_3_5", psave=False, cmap=True):
    path = os.path.join(file_dir, f"inputs/winds/HornsRev1/wind_{wind_name}.csv")
    name, bins = wind_name.split('_')[0], tuple(int(i) for i in wind_name.split('_')[-2:])
    if not os.path.exists(path):
        winds_pdf(bins, name=name, output="wind")
    winds = pd.read_csv(path, header=0)
    rad_i = winds.values[:, -1]
    barSlices = winds.values.shape[0]
    theta_i = np.linspace(0.0, 2 * np.pi, barSlices, endpoint=False)
    width = 2 * np.pi / barSlices

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_direction(-1), ax.set_theta_zero_location("N")
    if not cmap:
        colors = np.array(["k"] * barSlices)
        ax.bar(theta_i, rad_i, width=width, color=colors, bottom=np.min(rad_i) * 0.2,
               align="edge", alpha=0.8, )
    else:
        v_bin, _, _, _ = winds_discretization(bins)
        v_probs = weibull_cdf(np.repeat(v_bin[None, :], barSlices, axis=0),
                              np.repeat(winds.values[:, -2][:, None], len(v_bin), axis=1),
                              np.repeat(winds.values[:, -3][:, None], len(v_bin), axis=1))
        ratios = np.concatenate((np.ones((barSlices, 1)), v_probs[:, ::-1], ), axis=1)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(v_bin) + 1))[::-1]
        for i in range(len(v_bin) + 1):
            ax.bar(theta_i, rad_i * ratios[:, i], width=width, color=colors[i],
                bottom=np.min(rad_i) * 0.2, align="edge", edgecolor='k', lw=0.1,
                alpha=0.7, zorder=i)
        if name == 'average':
            ax.set_ylim([0, 0.025])

    ax.tick_params(labelsize=15, zorder=100)
    format_z = lambda x, pos: f'{x * 1e2:.1f}%'
    ax.yaxis.set_major_formatter(FuncFormatter(format_z))
    # ax.yaxis.set_major_formatter(PercentFormatter())
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax.grid(linestyle='--', linewidth=0.5, color='k', alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    if psave:
        plt.savefig(f"../params/rose_{wind_name}.png", format='png',
                    dpi=300, bbox_inches='tight')
        print("** Picture {} Save Done !** \n".format(psave))
    plt.show()


def winds_param(wd=None, pdf=None, bin_size=5, show=False):
    wind_pdf = Horns.wind_pdf if pdf is None else pdf
    angles, cs, ks, fs = wind_pdf[0, :], wind_pdf[1, :], wind_pdf[2, :], wind_pdf[3, :]
    samples = np.linspace(0, 330, int(360 / bin_size)) \
        if wd is None else wd
    # Scale parameters fitting
    spl_cs = splrep(angles, cs, k=2)
    inter_cs = splev(samples, spl_cs)
    # Shape parameters fitting
    spl_ks = splrep(angles, ks, k=2)
    inter_ks = splev(samples, spl_ks)
    # wind frequency fitting
    spl_fs = splrep(angles, fs, k=3)
    inter_fs = splev(samples, spl_fs)
    if (inter_fs < 0.).any():
        spl_fs = splrep(angles, fs, k=1)
        inter_fs = splev(samples, spl_fs)
    new_fs = inter_fs / np.sum(inter_fs)
    # print(new_fs)

    if show:
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(311)
        ax1.plot(samples, inter_cs, linestyle='--', c='r', lw=2., label='Spline')
        ax1.plot(angles, cs, linestyle='-', c='b', lw=2., label='Scale')
        ax1.legend(loc='upper left')

        ax2 = fig.add_subplot(312)
        ax2.plot(samples, inter_ks, linestyle='--', c='r', lw=2., label='Spline')
        ax2.plot(angles, ks, linestyle='-', c='b', lw=2., label='Shape')
        ax2.legend(loc='upper left')

        ax3 = fig.add_subplot(313)
        ax3.plot(samples, inter_fs, linestyle='--', c='r', lw=2., label='Spline')
        ax3.plot(angles, fs, linestyle='-', c='b', lw=2., label='Frequency')
        ax3.legend(loc='upper left')

        # plt.savefig("parameters/horns_winds_pdf.png", format='png',
        #             dpi=300, bbox_inches='tight')
        plt.show()

    return inter_cs, inter_ks, new_fs


def winds_dist_plot(name='horns', bins=(3, 10)):
    dist_params = winds_loader('wind', name, bins).values
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)
    plt.rc('font',family='Times New Roman')
    speeds = np.arange(0, 30, 0.5)
    colors = plt.cm.rainbow(np.linspace(0, 1, dist_params.shape[0]))
    labels = (dist_params[:, 0] + dist_params[:, 1]) / 2
    for i in range(dist_params.shape[0]):
        pdf = weibull_pdf(
            speeds, dist_params[i, 3], dist_params[i, 2])
        ax.plot(speeds, pdf,
                color=colors[i],
                label=labels[i],
                linewidth=1.,
                linestyle='-',
                )
    ax.set_xlim([0., 31.])
    ax.set_xlabel('Wind Speed (m/s)', ppt.font20)
    ax.set_xticks(np.arange(0, 31, 5))
    # axl.set_xticklabels(['5', '10', '15', '20',])
    ax.set_ylim([0, 0.12])
    ax.set_ylabel('Frequency', ppt.font20)
    ax.set_yticks(np.arange(0, 0.121, 0.02))
    ax.set_yticklabels(['', '0.02', '0.04', '0.06', '0.08', '0.10', '0.12'])
    ax.tick_params(labelsize=15, direction='in')
    # ax.legend(loc='upper right', prop=ppt.font20,
    #           edgecolor='None', frameon=False,
    #           labelspacing=0.4, bbox_transform=ax.transAxes,)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.savefig("../params/wind_dist.png", format='png',
                dpi=300, bbox_inches='tight')

    plt.show()


def winds_pdf(bins=(3, 5), speeds=(4, 25), pdf=None, name=None,
              show=False, psave=False, **kwargs):
    v_bin, v_point, w_bin, w_point = winds_discretization(bins, speeds=speeds)
    if name in ['single', 'average']:
        pdf = Horns.wind_pdf
        pdf[-1, :] = Horns.ref_pdf[name]
    cs, ks, fs = winds_param(wd=w_point, pdf=pdf, show=False)
    Nv, Nw = v_point.shape[0], w_point.shape[0]
    bivariate_pdf = np.zeros((Nv, Nw))
    for i in range(Nw):
        v_fs = weibull_cdf(v_bin[1:], cs[i], ks[i]) - \
            weibull_cdf(v_bin[:-1], cs[i], ks[i])
        bivariate_pdf[:, i] = v_fs * fs[i]

    if show:
        rcParams.update(ppt.config)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(w_point, v_point)
        surf = ax.plot_surface(x, y, bivariate_pdf, rstride=1, cstride=1,
                               cmap=plt.get_cmap('rainbow'))
        ax.contourf(x, y, bivariate_pdf, zdir='z', offset=-2)
        ax.view_init(elev=45, azim=60)
        # print('ax.azim {}'.format(ax.azim))
        # print('ax.elev {}'.format(ax.elev))
        # ax.set_xlim(-10, 370), ax.set_ylim(0, 26), ax.set_zlim(0, 3.5e-3)
        ax.set_xlabel(r'$\mathit{\theta}(^o)$', ppt.font20, labelpad=15)
        ax.set_ylabel(r'$\mathit{v(m/s)}$', ppt.font20, labelpad=15)
        ax.set_zlabel(r'$\mathit{p(v, \theta)}$', ppt.font20, labelpad=10)
        ax.set_xticks(np.arange(0, 361, 60))
        ax.set_yticks(np.arange(5, 26, 5))
        ax.tick_params(labelsize=15, colors='k', direction='in')
        # ax.tick_params(axis='z', which='major', pad=8)
        labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        # ax.grid(linestyle=':', linewidth=3, color='k', zorder=0)
        # ax.xaxis._axinfo["grid"]['color'] = "#ee0009"
        for xaxis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            xaxis._axinfo["grid"].update({'linewidth':1, 'linestyle':":"})

        format_z = lambda x, pos: '%.1f' % (x * 1e3)
        ax.zaxis.set_major_formatter(FuncFormatter(format_z))
        ax.text(370., 0., np.max(bivariate_pdf) * 0.95, r'$\mathit{×10^{-3}}$',
                color="k", fontsize=15, zorder=3, style ='italic', )

        ax_cbar = fig.add_axes([0.85, 0.2, 0.02, 0.5])
        fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.005, cax=ax_cbar)
        ax_cbar.yaxis.set_major_formatter(FuncFormatter(format_z))
        ax_cbar.tick_params(labelsize=15, colors='k', direction='in')
        ax.text(10., 30., np.max(bivariate_pdf) * 1.52, r'$\mathit{×10^{-3}}$',
                color="k", fontsize=15, zorder=3, style ='italic')

        # ax.ticklabel_format(style='sci', scilimits=(-4,-3), axis='z')
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
        if psave:
            plt.savefig(f"../params/pdf_{name}_{bins[0]}_{bins[1]}.png", format='png',
                        dpi=400, bbox_inches='tight')
        plt.show()

    name = name or 'horns'
    if kwargs.get("output", False) == "wind":
        winds = {}
        winds['l-1'], winds['theta_l-1'], winds['theta_l'] = \
            np.arange(w_point.shape[0]), w_bin[:-1], w_bin[1:]
        winds['k'], winds['c'], winds['w_l-1'] = \
            np.around(ks, 6), np.around(cs, 6), np.around(fs, 6)
        winds = pd.DataFrame(winds, )
        winds.to_csv(f'../params/wind_{name}_{bins[0]}_{bins[1]}.csv', index=False)

    if kwargs.get("output", False) == "pdf":
        pdf_data = pd.DataFrame(np.around(bivariate_pdf, 6), columns=w_point, index=v_point)
        pdf_data.to_csv(f'../params/pdf_{name}_{bins[0]}_{bins[1]}.csv', )

    return v_bin, v_point, w_point, bivariate_pdf


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class Horns(object):

    wind_pdf = np.array([[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
                            [8.89, 9.27, 8.23, 9.78, 11.64, 11.03, 11.50,
                                11.92, 11.49, 11.08, 11.34, 10.76, 8.89],
                            [2.09, 2.13, 2.29, 2.30, 2.67, 2.45,
                                2.51, 2.40, 2.35, 2.27, 2.24, 2.19, 2.09],
                            [4.82, 4.06, 3.59, 5.27, 9.12, 6.97, 9.17,
                            11.84, 12.41, 11.34, 11.70, 9.69, 4.82]])

    ref_pdf = {'single': np.array([[1.90, 1.90, 1.90, 1.90, 1.90, 1.90, 1.90,
                                    1.90, 79.10, 1.90, 1.90, 1.90, 1.90]]),
               'average': np.array([[8.33, 8.33, 8.33, 8.33, 8.33, 8.33, 8.33,
                                     8.33, 8.33, 8.33, 8.33, 8.33, 8.33]])}


def weibull_pdf(v, scale, shape):
    return (shape / scale) * (v / scale)**(shape - 1) * np.exp(-(v / scale) ** shape)


def weibull_cdf(v, scale, shape):
    return 1 - np.exp(-(v / scale) ** shape)


def winds_discretization(bins, speeds=(4, 25)):
    vbins, wbins, v_in, v_out = bins[0], bins[1], speeds[0], speeds[1]
    v_bin = np.append(np.arange(v_in, v_out, vbins), v_out)
    v_point = (v_bin[:-1] + v_bin[1:]) / 2
    w_point = - 0.5 * wbins + (np.arange(1, int(360 / wbins) + 1) - 0.5) * wbins
    w_bin = np.append((w_point - 0.5 * wbins), (w_point[-1] + 0.5 * wbins))
    return v_bin, v_point, w_bin, w_point


def winds_loader(data, name, bins, speed=(4, 25)):
    data_path = f"../params/{data}_{name}_{bins[0]}_{bins[1]}.csv"
    if not os.path.exists(data_path):
        winds_pdf(bins, speed, name=name, output=data)
    return pd.read_csv(data_path, header=0, index_col=0)



if __name__ == "__main__":
    print(file_dir)