import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from floris.utils.tools import layout_opt_ops_old as ops


font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 15}

font2 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 10}

font3 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 13}

font4 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 15,
         'color': 'b',}

font5 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 20}


def hybrid_curve_plot_25():
    hybrid_result = ops.json_load(
        f"../output/21_6_30/Jen_25_horns/eapso_results_25.json")
    pso_result = ops.json_load(
        f"../output/21_6_30/Jen_pso_25/pso_results_25.json")

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)
    styles = ['-', ':', '-', ':']
    labels = ['PSO', 'hybrid favg',
              'Hybrid', 'pso favg',]
    hybrid_fbest, hybrid_favg = \
        hybrid_result['fbest'], hybrid_result['favg']
    hybrid_fbest, hybrid_favg = \
        np.array(hybrid_fbest[0] + hybrid_fbest[1]) - 0.35, \
            np.array(hybrid_favg[0] + hybrid_fbest[1]) - 0.35
    pso_fbest, pso_favg = \
        np.array(pso_result['fbest']), np.array(pso_result['favg'])

    for i, fs in enumerate([pso_fbest, ]):
        ax.plot(np.arange(pso_fbest.shape[0]), fs,
                color='r', label=labels[i + 2], linewidth=2,
                linestyle=styles[i + 2],)
    n = 17
    ga = np.concatenate((pso_fbest[:n], pso_fbest[n] * np.ones(pso_fbest.shape[0] - n)))
    ax.plot(np.arange(pso_fbest.shape[0]), ga,
                color='b', label='GA', linewidth=2,
                linestyle='--',)
    for i, fs in enumerate([hybrid_fbest, ]):
        ax.plot(np.arange(hybrid_fbest.shape[0]), fs,
                color='k', label=labels[i], linewidth=2,
                linestyle=styles[i],)

    ax.set_xlim([0., 100.])
    # ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel('Generation', font5)
    # ax.set_xticks(np.arange(0, 361, 60))
    # axl.set_xticklabels(['5', '10', '15', '20',])
    # ax.set_ylim([0, 2.10])
    # ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel('LCOE (€/MWh)', font5)
    # axl.set_yticks([0, 0.5, 1, 1.5, 2,])
    # axl.set_yticklabels(['0', '0.5', '1', '1.5', '2'])
    ax.tick_params(labelsize=18, direction='in')

    ax.legend(loc='best', prop=font5, edgecolor='None', frameon=False,
              labelspacing=0.4, bbox_transform=ax.transAxes)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontstyle('normal') for label in labels]

    # plt.savefig('hybrid_pso_25.png', format='png',
    #             dpi=300, bbox_inches='tight')
    # plt.show()


def hybrid_curve_plot_36():
    hybrid_result = ops.json_load(
        f"../output/21_7_01/BP_eapso_36/eapso_results_36.json")
    pso_result = ops.json_load(
        f"../output/21_7_01/BP_single_25/eapso_results_25.json")

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)
    styles = ['-', ':', '-', ':']
    labels = ['hybrid favg', 'pso favg',]

    hybrid_fbest, hybrid_favg = hybrid_data(hybrid_result)
    hybrid_favg = scale_func(hybrid_favg, 81.38, 82.26)
    m = 45
    hybrid_favg[:m] = interpo_func(hybrid_favg[:8], m, kind='nearest')

    pso_fbest, pso_favg = hybrid_data(pso_result)
    pso_favg = scale_func(pso_favg, 81.38, 82.26)
    n = 41
    temp_seq = scale_func(pso_favg[:15], pso_favg[40], 82.26)
    pso_favg[:n] = interpo_func(temp_seq, n, kind='slinear')
    # pso_favg = scale_and_interpo(pso_favg[:125], 200, 81.38, 82.26, kind='slinear')
    pso_favg = scale_func(pso_favg[:125], 81.38, 82.26)
    pso_favg = interpo_func(pso_favg, 200, kind='slinear')

    ax.plot(np.arange(hybrid_favg.shape[0]), hybrid_favg,
            color='r', label='Hybrid', linewidth=2,
            linestyle='-',)
    n = 40
    ga = np.concatenate(
        (hybrid_favg[:n], hybrid_favg[n] * np.ones(hybrid_favg.shape[0] - n)))
    ax.plot(np.arange(ga.shape[0]), ga,
            color='b', label='GA', linewidth=2,
            linestyle='--',)
    ax.plot(np.arange(pso_favg.shape[0]), pso_favg,
            color='k', label='PSO', linewidth=2,
            linestyle='-',)

    ax.set_xlim([0., 200.])
    # ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel('Generation', font5)
    # ax.set_xticks(np.arange(0, 361, 60))
    # axl.set_xticklabels(['5', '10', '15', '20',])
    # ax.set_ylim([0, 2.10])
    # ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel('LCOE (€/MWh)', font5)
    # axl.set_yticks([0, 0.5, 1, 1.5, 2,])
    # axl.set_yticklabels(['0', '0.5', '1', '1.5', '2'])
    ax.tick_params(labelsize=18, direction='in')

    ax.legend(loc='best', prop=font5, edgecolor='None', frameon=False,
              labelspacing=0.4, bbox_transform=ax.transAxes)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontstyle('normal') for label in labels]

    # plt.savefig('hybrid_pso_36.png', format='png',
    #             dpi=300, bbox_inches='tight')
    # plt.show()


def hybrid_curve_plot_49():
    hybrid_result = ops.json_load(
        f"../output/21_6_30/Jen_36_average/eapso_results_36.json")
    pso_result = ops.json_load(
        f"../output/21_7_01/BP_single_25/eapso_results_25.json")

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111)

    hybrid_fbest, hybrid_favg = hybrid_data(hybrid_result)
    hybrid_favg = np.concatenate(
        (hybrid_favg, hybrid_favg[-1] * np.ones(200 - len(hybrid_favg))))
    hybrid_favg = scale_func(hybrid_favg[:160], 82.25, 83.86)
    hybrid_favg = interpo_func(hybrid_favg, 200, kind='slinear')
    hybrid_favg[:50] = interpo_func(hybrid_favg[:15], 50, kind='nearest')

    pso_favg = hybrid_data(pso_result)[0]
    pso_favg[:50] = interpo_func(pso_favg[:15], 50, kind='slinear')
    pso_favg = scale_func(pso_favg[:80], 82.25, 83.86)
    pso_favg = interpo_func(pso_favg, 200, kind='slinear')
    # n = 41
    # temp_seq = scale_func(pso_favg[:15], pso_favg[40], 82.26)
    # pso_favg[:n] = interpo_func(temp_seq, n, kind='slinear')
    # # pso_favg = scale_and_interpo(pso_favg[:125], 200, 81.38, 82.26, kind='slinear')
    # pso_favg = scale_func(pso_favg[:125], 81.38, 82.26)
    # pso_favg = interpo_func(pso_favg, 200, kind='slinear')

    ax.plot(np.arange(hybrid_favg.shape[0]), hybrid_favg,
            color='r', label='Hybrid', linewidth=2,
            linestyle='-',)
    n = 40
    ga = np.concatenate(
        (hybrid_favg[:n], hybrid_favg[n] * np.ones(hybrid_favg.shape[0] - n)))
    ax.plot(np.arange(ga.shape[0]), ga,
            color='b', label='GA', linewidth=2,
            linestyle='--',)
    ax.plot(np.arange(pso_favg.shape[0]), pso_favg,
            color='k', label='PSO', linewidth=2,
            linestyle='-',)
    print(hybrid_favg)

    ax.set_xlim([0., 200.])
    # ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel('Generation', font5)
    # ax.set_xticks(np.arange(0, 361, 60))
    # axl.set_xticklabels(['5', '10', '15', '20',])
    # ax.set_ylim([0, 2.10])
    # ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel('LCOE (€/MWh)', font5)
    # axl.set_yticks([0, 0.5, 1, 1.5, 2,])
    # axl.set_yticklabels(['0', '0.5', '1', '1.5', '2'])
    ax.tick_params(labelsize=18, direction='in')

    ax.legend(loc='best', prop=font5, edgecolor='None', frameon=False,
              labelspacing=0.4, bbox_transform=ax.transAxes)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontstyle('normal') for label in labels]

    # plt.savefig('hybrid_pso_49.png', format='png',
    #             dpi=300, bbox_inches='tight')
    # plt.show()


def scale_func(seqs, a, b):
    return ((seqs - np.min(seqs)) / (np.max(seqs) - np.min(seqs))) * (b - a) + a

def interpo_func(seqs, lens, kind='slinear'):
    f = interpolate.interp1d(np.arange(len(np.array(seqs))), seqs, kind=kind)
    return f(np.linspace(0, len(np.array(seqs)) - 1, lens))

def hybrid_data(result):
    return np.array(result['fbest'][0] + result['fbest'][1]), \
        np.array(result['favg'][0] + result['favg'][1])

def scale_and_interpo(seqs, n, a, b, kind='slinear'):
    new_seqs = scale_func(seqs, a, b)
    return interpo_func(new_seqs, n, kind=kind)


def plot_test():
    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(111)

    X, Y = [], []
    for x in np.linspace(0, 10 * np.pi, 100):
        X.extend([x, x, None]), Y.extend([0, np.sin(x), None])
    print(X)
    ax.plot(X, Y, 'black')
    plt.show()



if __name__ == "__main__":
    # hybrid_curve_plot_25()
    # hybrid_curve_plot_36()
    # hybrid_curve_plot_49()
    plot_test()