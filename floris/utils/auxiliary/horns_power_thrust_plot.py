import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import rcParams
# from matplotlib.ticker import FuncFormatter
# from scipy.interpolate import splev, splrep, interp1d

from floris.utils.modules.tools import plot_property as ppt

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                       MAIN                                   #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def curve_show(curves):
    fit_power, fit_ct = curves
    vel = np.arange(4.5, 21., 0.5)
    fit_power_y, fit_ct_y = np.vectorize(fit_power)(vel), np.vectorize(fit_ct)(vel)
    power_y, ct_y = \
        np.loadtxt("../params/horns1_power.txt")[4:], \
            np.loadtxt("../params/horns1_thrust.txt")[5:-8]
    
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    
    plt.rc('font',family='Times New Roman')
    lns1 = ax1.plot(vel, fit_power_y,
                    c='k',
                    lw=4.,
                    linestyle='-',
                    zorder=3,
                    label='Fitting: P',)
    lns2 = ax1.plot(power_y[:, 0], power_y[:, 1],
                    c='w',
                    lw=0.00,
                    zorder=1,
                    label='Manufacturer: P',
                    markersize=16,
                    marker='o',
                    markeredgecolor='k',
                    markeredgewidth=3.)
    ax1.set_xlim([3.5, 21.5])
    ax1.set_xlabel('Wind Speed (m/s)', ppt.font25)
    # ax1.set_xlabel(r'$\mathcal{Wind Speed}(m/s^2)$', ppt.font25)
    ax1.set_xticks([5, 10, 15, 20, ])
    ax1.set_xticklabels(['5', '10', '15', '20',])
    ax1.set_ylim([0, 2.10])
    ax1.set_ylabel('Power (MW)', ppt.font25)
    ax1.set_yticks([0, 0.5, 1, 1.5, 2,])
    ax1.set_yticklabels(['0', '0.5', '1', '1.5', '2'])
    ax1.tick_params(labelsize=20, direction='in')
    ax1.grid(linestyle=':', linewidth=1.5, color='k', zorder=0)
    
    ax2 = ax1.twinx()  # this is the important function
    lns3 = ax2.plot(vel, fit_ct_y,
                    c='b',
                    lw=4.,
                    linestyle='--',
                    zorder=3,
                    label='Fitting: Ct',)
    lns4 = ax2.plot(ct_y[:, 0], ct_y[:, 1],
                    c='w',
                    lw=0.00,
                    zorder=1,
                    label='Manufacturer: Ct',
                    markersize=12,
                    marker='x',
                    markeredgecolor='b',
                    markeredgewidth=3.)
    ax2.set_ylim([0, 1.05])
    ax2.set_ylabel('Thrust Coefficient', ppt.font25b)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.,])
    ax2.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax2.tick_params(labelsize=20, colors='b', direction='in')
    ax2.spines['right'].set_color('b')
    
    # added these three lines
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right', prop=ppt.font22,
                edgecolor='None', frameon=False, labelspacing=0.4,
                bbox_transform=ax1.transAxes, bbox_to_anchor=(1.0, 0.7))
    
    labels = ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # plt.savefig("../params/p_ct_curve.png", format='png',
    #             dpi=300, bbox_inches='tight')
    plt.show()



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class Horns_rev_1(object):
    
    @classmethod
    def pow_curve(cls, vel):
        if vel <= 4.:
            return 0.
        elif vel >= 15.:
            return 2.
        else:
            return 1.45096246e-07 * vel**8 - 1.34886923e-05 * vel**7 + \
                5.23407966e-04 * vel**6 - 1.09843946e-02 * vel**5 + \
                    1.35266234e-01 * vel**4 - 9.95826651e-01 * vel**3 + \
                        4.29176920e+00 * vel**2 - 9.84035534e+00 * vel + \
                            9.14526132e+00

    @classmethod
    def ct_curve(cls, vel):
        if vel <= 10.:
            vel = 10.
        elif vel >= 20.:
            vel = 20.
        return np.array([-2.98723724e-11, 5.03056185e-09, -3.78603307e-07,  1.68050026e-05,
                            -4.88921388e-04,  9.80076811e-03, -1.38497930e-01,  1.38736280e+00,
                            -9.76054549e+00,  4.69713775e+01, -1.46641177e+02,  2.66548591e+02,
                            -2.12536408e+02]).dot(np.array([vel**12, vel**11, vel**10, vel**9,
                                                            vel**8, vel**7, vel**6, vel**5,
                                                            vel**4, vel**3, vel**2, vel, 1.]))



if __name__ == "__main__":
    power, thrust = Horns_rev_1.pow_curve, Horns_rev_1.ct_curve,
    curve_show((power, thrust))
    # print(file_dir)