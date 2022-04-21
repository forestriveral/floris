import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator

from floris.tools import FlorisInterface
from floris.tools import cut_plane
from floris.utils.modules.control.real_yaw_simulator import YawSimulator
from floris.utils.visual import property as ppt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                               DYNAMIC_YAW_PLOT                               #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def dynamic_power_plot(export=False):
    yawer = YawSimulator('../inputs/yaw_test_1.json', '../inputs/winds/201301010930.xlsx',
                         results='../outputs/yaw_opt_results.csv')
    no_wake, _ = yawer.power_calculation(np.zeros(len(yawer.wd)), no_wake=True)
    wd, ws = yawer.results['wd'].values, yawer.results['ws'].values
    baseline_power, power = yawer.results['baseline_power'].values / no_wake, \
        yawer.results['power'].values / no_wake
    # baseline_yaw, yaw = yawer.results['baseline_yaw'].values, \
    #     yawer.results[yawer.get_data('control', 'yaw')].values
    # D = yawer.fi.floris.farm.turbine_map.turbines[0].rotor_diameter
    time = np.arange(len(wd)) * yawer.delt
    
    # define some empty list for line plot
    bpx, bpy = [], []
    px, py = [], []
    bpmean, pmean = [], []
    wdx, wdy = [], []
    wsx, wsy = [], []
    # create the figure
    fig = plt.figure(figsize=(10, 12), dpi=150)
    
    # the baseline power and controlled power lines
    axp = fig.add_subplot(212)
    bpline, = axp.plot(bpx, bpy, c="k", lw=2., label='Baseline')
    pline, = axp.plot(px, py, c="r", lw=2., label='Controlled')
    bpmean, = axp.plot(bpx, bpmean, ls='--', c='k', lw=2., alpha=0.7)
    pmean, = axp.plot(px, pmean, ls='--', c='r', lw=2., alpha=0.7)
    
    # the wind direction line
    axwd = fig.add_subplot(411)
    wdline, = axwd.plot(wdx, wdy, c="k", lw=2., label='Wind Speed')
    
    # the wind speed line
    axws = fig.add_subplot(412)
    wsline, = axws.plot(wsx, wsy, c="k", lw=2., label='Wind Direction')
    
    def init():
        # the baseline power and controlled power lines settings
        axp.set_xlabel('Time (min)', ppt.font15)
        axp.set_xlim(0, len(wd))
        stamp = np.arange(0, np.max(axp.get_xticks() * yawer.delt), 900)
        axp.set_xticks(stamp)
        axp.set_xticklabels([f'{t / 60:.0f}' for t in stamp])
        axp.set_ylabel('Normalized Power', ppt.font15)
        axp.set_ylim((0.4, 1.1))
        # axp.set_yticks([0, 0.5, 1, 1.5, 2,])
        # axp.set_yticklabels(['0', '0.5', '1', '1.5', '2'])
        axp.yaxis.set_major_locator(MultipleLocator(0.2))
        axp.tick_params(labelsize=15, colors='k', direction='in',
                        top=True, bottom=True, left=True, right=True)
        
        # the wind direction line settings
        axwd.axhline(y=np.mean(wd), color='k', alpha=0.5,
                    linestyle='--', linewidth=2)
        axwd.set_xlim(0, len(wd))
        axwd.set_xticks(stamp)
        # axwd.set_xticklabels([f'{t / 60:.0f}' for t in stamp])
        axwd.set_xticklabels([])
        axwd.set_ylim((np.min(wd) * 0.85, np.max(wd)* 1.15))
        axwd.set_ylabel('Degree ($^o$)', ppt.font15)
        axwd.yaxis.set_major_locator(MultipleLocator(20.))
        axwd.tick_params(labelsize=15, colors='k', direction='in',
                        bottom=True, left=True)
        
        # the wind speed line settings
        axws.axhline(y=np.mean(ws), color='k', alpha=0.5,
                    linestyle='--', linewidth=2)
        axws.set_xlim(0, len(ws))
        # axws.set_xlabel(f'Time (min)', ppt.font15)
        axws.set_xticks(stamp)
        # axws.set_xticklabels([f'{t / 60:.0f}' for t in stamp])
        axws.set_xticklabels([])
        axws.set_ylim((np.min(ws) * 0.85, np.max(ws)* 1.15))
        axws.set_ylabel('Speed (m/s)', ppt.font15)
        axws.yaxis.set_major_locator(MultipleLocator(2.))
        axws.tick_params(labelsize=15, colors='k', direction='in',
                        bottom=True, left=True)
        
        tick_labs = axp.get_xticklabels() + axp.get_yticklabels() + \
            axwd.get_xticklabels() + axwd.get_yticklabels() + \
                axws.get_xticklabels() + axws.get_yticklabels()
        [tick_lab.set_fontname('Times New Roman') for tick_lab in tick_labs]
        
        fig.subplots_adjust(hspace=0.1)
        fig.tight_layout()
        
        return bpline, pline, bpmean, pmean, wdline, wsline

    def update(step):
        # the baseline power and controlled power lines update
        bpx.append(step * yawer.delt); bpy.append(baseline_power[step])
        px.append(step * yawer.delt); py.append(power[step])
        bpline.set_data(bpx, bpy); pline.set_data(px, py)
        
        # the mean baseline power and controlled power lines update
        bpmean.set_data(time, np.mean(bpy) * np.ones(len(time)))
        pmean.set_data(time, np.mean(py) * np.ones(len(time)))
        bpmean.set_label(f'Baseline mean ({np.mean(bpy):.3f})')
        pmean.set_label(f'Controlled mean ({np.mean(py):.3f})')
        
        # the wind direction and speed lines update
        wdx.append(step * yawer.delt); wdy.append(wd[step])
        wsx.append(step * yawer.delt); wsy.append(ws[step])
        wdline.set_data(wdx, wdy); wsline.set_data(wsx, wsy)
        
        for ax in [axp, axwd, axws]:
            ax.legend(loc="best", prop=ppt.font13, edgecolor='None',
                      frameon=False, labelspacing=0.4, bbox_transform=ax.transAxes)
        
        return bpline, pline, bpmean, pmean, wdline, wsline
    
    
    ani = FuncAnimation(fig, update, frames=len(wd), init_func=init,
                        interval=120, repeat=False)
    if export:
        ani.save("../outputs/power_test.gif", writer='pillow', fps=60, dpi=100)
    # plt.show()


def dynamic_turbine_plot(export=False):
    yawer = YawSimulator('../inputs/yaw_test_1.json', '../inputs/winds/201301010930.xlsx',
                         results='../outputs/yaw_opt_results.csv')
    wd, ws = yawer.results['wd'].values, yawer.results['ws'].values
    baseline_yaw, yaw = yawer.results['baseline_yaw'].values, \
        yawer.results[yawer.get_data('control', 'yaw')].values
    D = yawer.fi.floris.farm.turbine_map.turbines[0].rotor_diameter
    # time = np.arange(len(wd)) * yawer.delt
    
    fig = plt.figure(dpi=150)
    minSpeed, maxSpeed, bpx1, bpx2, bpZm = plane_data(
        plane_plot(yawer.fi, wd[0], ws[0], -1 * baseline_yaw[0]))
    names = globals()
    plot_cmp = 'bwr'
    # plot_cmp = 'coolwarm'
    
    # baseline turbine control plot
    ax = fig.add_subplot(211)
    for i in range(yawer.num_t):
        names[f'bturbine_{i}'], = ax.plot([], [], c='k', lw='1.5')
    ax.set_xticklabels([])
    
    # controlled turbine control plot
    ax1 = fig.add_subplot(212)
    for i in range(yawer.num_t):
        names[f'cturbine_{i}'], = ax1.plot([], [], c='k', lw='1.5')
    
    # time point added
    time_text = ax.text(0.01, 1.2, '', va='top', ha='left', fontdict=ppt.font10,
                        transform=ax.transAxes)
    
    # initialized status
    def init():
        # baseline turbine settings
        yawer.fi.floris.farm.set_yaw_angles(yaw_angles=float(-baseline_yaw[0]))
        loc_x0, loc_y0 = turbine_plot(yawer.fi.layout_x, yawer.fi.layout_y,
                                      yawer.fi.get_yaw_angles(), D, wind_direction=wd[0])
        for axes, label, prefix in zip([ax, ax1], ['(a)Baseline', '(b)Controlled'], ['b', 'c']):
            for i in range(yawer.num_t):
                exec(f"{prefix}turbine_{i}.set_data(loc_x0[{i}], loc_y0[{i}])")
            axes.pcolormesh(bpx1, bpx2, bpZm, cmap=plot_cmp, vmin=minSpeed, vmax=maxSpeed,
                            shading="nearest")
            axes.text(0.01, 0.95, label, va='top', ha='left', fontdict=ppt.font10,
                      transform=axes.transAxes)
            axes.yaxis.set_major_locator(MultipleLocator(1.5))
            axes.tick_params(labelsize=10, colors='k', direction='in', bottom=True, left=True)
            ticks = axes.get_xticklabels() + axes.get_yticklabels() 
            [tick.set_fontname('Times New Roman') for tick in ticks]
            axes.set_aspect("equal")
        
        fig.subplots_adjust(hspace=-0.4)
        fig.tight_layout()

    # plot update function
    def update(step):
        # print(step)
        zip_data = zip([ax, ax1], [float(-baseline_yaw[step]), list(-yaw[step])], ['b', 'c'])
        for axes, offset, prefix in zip_data:
            yawer.fi.floris.farm.set_yaw_angles(yaw_angles=offset)
            loc_x, loc_y = turbine_plot(yawer.fi.layout_x, yawer.fi.layout_y,
                                        yawer.fi.get_yaw_angles(), D,
                                        wind_direction=wd[step])
            for i in range(yawer.num_t):
                exec(f"{prefix}turbine_{i}.set_data(loc_x[{i}], loc_y[{i}])")
            minSpeed, maxSpeed, _, _, Zm = plane_data(
                plane_plot(yawer.fi, wd[step], ws[step], offset))
            axes.pcolormesh(bpx1, bpx2, Zm, cmap=plot_cmp, vmin=minSpeed,
                            vmax=maxSpeed, shading="nearest")
        
        time_point = step * yawer.delt if step * yawer.delt / 60 < 1. else step * yawer.delt / 60
        time_unit = 's' if step * yawer.delt / 60 < 1. else 'min'
        time_text.set_text(
            f'Time = {time_point:.1f} {time_unit}    Wind Direction = {wd[step]:.1f} degree')
    
    
    ani = FuncAnimation(fig, update, frames=len(wd[:900]), init_func=init,
                        interval=120, repeat=False)
    
    if export:
        ani.save("../outputs/turbine_test.gif", writer='imagemagick', fps=60, dpi=120)
    # plt.show()


def plane_data(plane):
    minSpeed = plane.df.u.min()
    maxSpeed = plane.df.u.max()
    x1_mesh = plane.df.x1.values.reshape(
        plane.resolution[1], plane.resolution[0])
    x2_mesh = plane.df.x2.values.reshape(
        plane.resolution[1], plane.resolution[0])
    u_mesh = plane.df.u.values.reshape(
        plane.resolution[1], plane.resolution[0])
    Zm = np.ma.masked_where(np.isnan(u_mesh), u_mesh)
    return minSpeed, maxSpeed, x1_mesh, x2_mesh, Zm


def plane_plot(fi, wd, ws, yaw):
    fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
    fi.floris.farm.set_yaw_angles(yaw_angles=yaw)
    hor_plane = fi.get_hor_plane(x_resolution=600, y_resolution=300,)
    D = fi.floris.farm.turbine_map.turbines[0].rotor_diameter
    return cut_plane.rescale_axis(hor_plane, x1_factor=D, x2_factor=D)


def turbine_plot(layout_x, layout_y, yaw_angles, D,
                 color=None, wind_direction=270.0):
    yaw_angles = np.array(yaw_angles) - wind_direction - 270
    if color is None:
        color = "k"
    loc_x, loc_y = [], []
    for x, y, yaw in zip(layout_x, layout_y, yaw_angles):
        R = D / 2.0 * 1.2
        x_0 = (x + np.sin(np.deg2rad(yaw)) * R) / D
        x_1 = (x - np.sin(np.deg2rad(yaw)) * R) / D
        y_0 = (y - np.cos(np.deg2rad(yaw)) * R) / D
        y_1 = (y + np.cos(np.deg2rad(yaw)) * R) / D
        # ax.plot([x_0, x_1], [y_0, y_1], color=color)
        loc_x.append([x_0, x_1])
        loc_y.append([y_0, y_1])
    return loc_x, loc_y



if __name__ == "__main__":
    # dynamic_power_plot()
    dynamic_turbine_plot(export=True)

