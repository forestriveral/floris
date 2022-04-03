import copy
import numpy as np
import matplotlib.pyplot as plt

from floris.tools import FlorisInterface
from floris.utils.visual import property as ppt
from floris.utils.modules.control.yaw_simulator import YawSimulator as YS


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                               YAW_OPTIMIZATION                               #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def time_history_plot(wd, ws, step, save=None):
    fig = plt.figure(figsize=(9, 6), dpi=120)
    ax1 = fig.add_subplot(211)
    time = np.arange(len(wd)) * step
    
    ax1.plot(time, wd,
             label='Wind Direction',
             color='k',
             linestyle='-',
             linewidth=1.5,
             markersize=0)
    ax1.axhline(y=np.mean(wd), color='k', alpha=0.5,
                linestyle='--', linewidth=2)
    # end_point = np.max(ax1.get_xticks())
    stamp = np.arange(0, np.max(ax1.get_xticks()), 300)
    ax1.set_xticks(stamp)
    ax1.set_xticklabels([f'{t / 60:.0f}' for t in stamp])
    ax1.set_ylabel('Degree ($^o$)', ppt.font15)
    ax1.tick_params(labelsize=15, colors='k', direction='in',
                    bottom=True, left=True)
    
    ax2 = fig.add_subplot(212)
    ax2.plot(time, ws,
             label='Wind Speed',
             color='k',
             linestyle='-',
             linewidth=1.5,
             markersize=0)
    ax2.axhline(y=np.mean(ws), color='k', alpha=0.5,
                linestyle='--', linewidth=2)
    ax2.set_ylabel('Speed (m/s)', ppt.font15)
    ax2.set_xlabel(f'Time (min)', ppt.font15)
    ax2.set_xticks(stamp)
    ax2.set_xticklabels([f'{t / 60:.0f}' for t in stamp])
    ax2.tick_params(labelsize=15, colors='k', direction='in',
                    bottom=True, left=True)
    
    tick_labs = ax1.get_xticklabels() + ax1.get_yticklabels() + \
        ax2.get_xticklabels() + ax2.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    
    if save:
        plt.savefig(f"../outputs/{save}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def yaw_baseline_plot(simulator, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(9, 6), dpi=120)
        ax = fig.add_subplot(111)
        
    cols = simulator.results.columns[1:5]
    labels = ['Origin', 'Target', 'Turbine', 'Yaw offset']
    colors = ['k', 'r', 'g', 'b']
    lines = ['-', ':', '--', '-']
    lns = []
    for i, col in enumerate(cols[:-1]):
        lns += ax.plot(np.arange(len(simulator.wd)) * simulator.delt,
                        simulator.results[col].values,
                        label=labels[i],
                        color=colors[i],
                        linestyle=lines[i],
                        linewidth=2,
                        markersize=0)
    ax.set_xlabel(r'Time ($s$)', ppt.font15)
    # ax.set_xlim((0., data.shape[1] + 1.))
    # ax.set_xticks()
    # ax.set_xticklabels([f'{i * simulator.delt}' for i in np.arange(len(simulator.wd))])
    ax.set_ylabel(r'Degree ($^o$)', ppt.font15)
    # ax.set_ylim((np.round(np.min(normal_array_power) * 0.8, 1), 1.1))
    ax.tick_params(labelsize=15, colors='k', direction='in',
                    top=True, bottom=True, left=True, right=True)
    
    axr = ax.twinx()
    lns += axr.plot(np.arange(len(simulator.wd)) * simulator.delt,
                    simulator.results[cols[-1]].values,
                    color=colors[-1],
                    label=labels[-1],
                    linestyle=lines[-1],
                    linewidth=2)
    # axr.set_ylim([-0.1, 1.1])
    axr.set_ylabel(r'Yaw offset ($^o$)', ppt.font15b)
    # axr.set_yticks([0, 0.25, 0.5, 0.75, 1.,])
    # axr.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
    axr.tick_params(labelsize=15, colors='b', direction='in')
    axr.spines['right'].set_color('b')
    
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels() + axr.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    
    ax.legend(lns, labels, loc="upper left", prop=ppt.font15, edgecolor='None',
                frameon=False, labelspacing=0.4, bbox_transform=ax.transAxes)
    
    # plt.savefig(f"../output/{}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def yaw_control_plot(simulator, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(9, 6), dpi=120)
        ax = fig.add_subplot(111)
        
    cols = simulator.results.columns[1:5]
    labels = ['Origin', 'Target', 'Turbine', 'Yaw offset']
    colors = ['k', 'r', 'g', 'b']
    lines = ['-', ':', '--', '-']
    lns = []
    for i, col in enumerate(cols[:-1]):
        lns += ax.plot(np.arange(len(simulator.wd)) * simulator.delt,
                        simulator.results[col].values,
                        label=labels[i],
                        color=colors[i],
                        linestyle=lines[i],
                        linewidth=2,
                        markersize=0)
    ax.set_xlabel(r'Time ($s$)', ppt.font15)
    # ax.set_xlim((0., data.shape[1] + 1.))
    # ax.set_xticks()
    # ax.set_xticklabels([f'{i * simulator.delt}' for i in np.arange(len(simulator.wd))])
    ax.set_ylabel(r'Degree ($^o$)', ppt.font15)
    # ax.set_ylim((np.round(np.min(normal_array_power) * 0.8, 1), 1.1))
    ax.tick_params(labelsize=15, colors='k', direction='in',
                    top=True, bottom=True, left=True, right=True)
    
    axr = ax.twinx()
    lns += axr.plot(np.arange(len(simulator.wd)) * simulator.delt,
                    simulator.results[cols[-1]].values,
                    color=colors[-1],
                    label=labels[-1],
                    linestyle=lines[-1],
                    linewidth=2)
    # axr.set_ylim([-0.1, 1.1])
    axr.set_ylabel(r'Yaw offset ($^o$)', ppt.font15b)
    # axr.set_yticks([0, 0.25, 0.5, 0.75, 1.,])
    # axr.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
    axr.tick_params(labelsize=15, colors='b', direction='in')
    axr.spines['right'].set_color('b')
    
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels() + axr.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    
    ax.legend(lns, labels, loc="upper left", prop=ppt.font15, edgecolor='None',
                frameon=False, labelspacing=0.4, bbox_transform=ax.transAxes)
    
    # plt.savefig(f"../output/{}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def yaw_power_plot(powers, wd, step, no_wake=None, save=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(15, 6), dpi=120)
        ax = fig.add_subplot(111)
    
    cols = ['baseline_power', 'power']
    labels = ['Baseline', 'Controlled', ]
    colors = ['k', 'r',]
    lines = ['-', '-', ]
    lns = []
    no_wake = 1. if no_wake is None else no_wake
    for i, col in enumerate(cols):
        # print(simulator.results[col].values)
        time = np.arange(len(wd)) * step
        power = powers[i] / no_wake
        lns += ax.plot(time, power,
                       label=labels[i],
                       color=colors[i],
                       linestyle=lines[i],
                       linewidth=1.5,
                       markersize=0)
        hln = ax.axhline(y=np.mean(power), color=colors[i], linestyle='--',
                         label=f'{labels[i]} mean ({np.mean(power):.3f})',
                         linewidth=2)
    ax.set_xlabel('Time (min)', ppt.font15)
    stamp = np.arange(0, np.max(ax.get_xticks()), 300)
    ax.set_xticks(stamp)
    ax.set_xticklabels([f'{t / 60:.0f}' for t in stamp])
    ax.set_ylabel('Total Power (MW)', ppt.font15)
    ax.set_ylim((np.round(np.min(powers / no_wake) * 0.9, 1),
                 np.round(np.max(powers / no_wake) * 1.1, 1)))
    ax.tick_params(labelsize=15, colors='k', direction='in',
                    top=True, bottom=True, left=True, right=True)
    
    tick_labs = ax.get_xticklabels() + ax.get_yticklabels()
    [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
    
    # lines = lines[0] + lines[1]+ lines[2] + lines[3]
    # labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, loc="lower left", prop=ppt.font15, edgecolor='None',
    #           frameon=False, labelspacing=0.4, bbox_transform=ax.transAxes)
    plt.legend(loc="best", prop=ppt.font15, edgecolor='None',
              frameon=False, labelspacing=0.4, bbox_transform=ax.transAxes)
    if save:
        plt.savefig(f"../outputs/{save}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def yaw_turbine_plot(fi, wd, ws, yaw, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(9, 6), dpi=150)
        ax = fig.add_subplot(111)
    
    D = fi.floris.farm.turbine_map.turbines[0].rotor_diameter
    fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
    fi.calculate_wake(yaw_angles=yaw)
    hor_plane = fi.get_hor_plane(x_resolution=400, y_resolution=300,
                                 # x_bounds=[-150.0, 150.0],
                                 # y_bounds=[-200.0, 200.0],
                                 )
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    wfct.visualization.plot_turbines(ax, fi.layout_x, fi.layout_y,
                                    fi.get_yaw_angles(), D,
                                    wind_direction=wd)
    
    # plt.savefig(f"../output/{}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def optimal_yaw_turbine_plot(fi, wd, ws, yaws, ax=None):
    optimal_fi = copy.deepcopy(fi)
    fig = plt.figure(dpi=150)
    plot_cmp = 'bwr'
    # plot_cmp = 'coolwarm'
    D = fi.floris.farm.turbine_map.turbines[0].rotor_diameter
    optimal_fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
    
    # baseline turbine control plot
    ax = fig.add_subplot(211)
    optimal_fi.calculate_wake(yaw_angles=yaws[0])
    hor_plane = optimal_fi.get_hor_plane(x_resolution=400, y_resolution=300,
                                 # x_bounds=[-150.0, 150.0],
                                 # y_bounds=[-200.0, 200.0],
                                 )
    wfct.visualization.visualize_cut_plane(hor_plane, cmap=plot_cmp, ax=ax)
    wfct.visualization.plot_turbines(ax, optimal_fi.layout_x,
                                     optimal_fi.layout_y,
                                     optimal_fi.get_yaw_angles(), D,
                                     wind_direction=wd)

    # controlled turbine control plot
    ax1 = fig.add_subplot(212)
    optimal_fi.calculate_wake(yaw_angles=yaws[1])
    hor_plane = optimal_fi.get_hor_plane(x_resolution=400, y_resolution=300,
                                 # x_bounds=[-150.0, 150.0],
                                 # y_bounds=[-200.0, 200.0],
                                 )
    wfct.visualization.visualize_cut_plane(hor_plane, cmap=plot_cmp, ax=ax1)
    wfct.visualization.plot_turbines(ax1, optimal_fi.layout_x,
                                     optimal_fi.layout_y,
                                     optimal_fi.get_yaw_angles(), D,
                                     wind_direction=wd)

    # time point added
    time_text = ax.text(0.01, 1.2, '', va='top', ha='left', fontdict=ppt.font10,
                        transform=ax.transAxes)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def time_formator(seqs):
    if np.max(seqs) < 600:
        return seqs, "s"
    else:
        return seqs / 60, "min"



if __name__ == "__main__":
    pass