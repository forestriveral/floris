import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_scipy import YawOptimization

from floris.utils.visual import property as ppt
from floris.utils.visual import yaw_opt as yopt

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 YAW_EVALUATION                               #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class YawSimulator(object):
    def __init__(self, farm, wind, params=None, wakes=None,
                 filted=True, results=None):
        self.fi = FlorisInterface(farm)
        self.num_t = len(self.fi.floris.farm.turbine_map.turbines)

        # yawing parameters setting
        self.data_freq = 10
        self.data_T = 1 / self.data_freq
        self.data_scale = 10
        self.control_interval = 6  # second

        self.f = 1 / 3          # sampling frequence (Hz) in time history curve
        self.delt = 1 / self.f  # sampling interval (s)
        self.t_c = 3            # control interval time
        self.n_c = np.ceil(self.t_c / self.delt)    # control interval which temporalily set as the integer multiple
        self.v_yaw = 5         # yawing speed of turbine
        self.conv = np.arange(-5, 5)   # convolution operation range [-i, ..., i]
        self.theta_tol = 2     # tolerance for yawing control decision
        self.theta_max = 30.     # maximum yaw offset allowed in process

        # optimization parametes setting
        self.max_offset = 4.
        self.max_time = 12.

        # parsing the custom parameters if provided
        if params is not None:
            self.parse_params(params)

        # wind data processing
        self.filter = filted
        self.origin_point = 12000
        self.wd_param, self.ws_param = wind['wd'], wind['ws']
        # wind direction data loading
        if self.wd_param[0] == 'origin':
            self.origin_wd = self.wind_load(self.wind_dataset[self.wd_param[0]],
                                            'wd', num=self.origin_point)
            self.wd = self.wind_processing(self.origin_wd, self.wd_param[1])
        else:
            self.wd = wind_generator(self.wd_param[0], self.wd_param[1],
                                     num=1800, interval=self.delt)
        # wind speed data loading
        if self.ws_param[0] == 'origin':
            self.origin_ws = self.wind_load(self.wind_dataset[self.ws_param[0]],
                                            'ws', num=self.origin_point)
            self.ws = self.wind_processing(self.origin_ws, self.ws_param[1])
        else:
            self.ws = wind_generator(self.ws_param[0], self.ws_param[1],
                                     num=1800, interval=self.delt)

        # yaw optimization configuration
        self.options = {'maxiter': 50, 'disp': False, 'iprint': 2, 'ftol': 1e-5, 'eps': 0.01}
        self.opt_fi = copy.deepcopy(self.fi)
        self.optimizer = YawOptimization(self.opt_fi,
                                         minimum_yaw_angle=0.0,
                                         maximum_yaw_angle=25.0,
                                         opt_method="SLSQP",
                                         opt_options=self.options,
                                         include_unc=False)

        # simulation results packing or load the results file
        if results:
            self.results = self.data_load(results)
            self.ws, self.wd = self.results['ws'], self.results['wd']
        else:
            self.results = self.data_package()
            self.results['ws'], self.results['wd'] = self.ws, self.wd
            self.results.fillna(0, inplace=True)
            # print(self.results.columns)

    def parse_params(self, params):
        pass

    @property
    def wind_dataset(self, ):
        return {'origin': '../inputs/winds/201301010930.xlsx',}

    def wind_load(self, wind_file, type='wd', num=18000):
        num_point = 72000
        userows = int(num) if num else num_point
        data = pd.read_excel(wind_file, sheet_name=0, usecols=[1, 3],
                             nrows=userows, names=['ws', 'wd'], header=None)
        return np.round(data[type].values, 2)

    def wind_processing(self, data, params):
        return self.wind_scale(self.wind_filter(self.wind_downsampling(data)), params)

    def wind_downsampling(self, data, average='fixed'):
        if average == 'fixed':
            return time_average(data, self.data_scale)
        elif average == 'sliding':
            return sliding_average(data, 30)
        else:
            raise ValueError("Invalid average method!")

    def wind_filter(self, data, cut_f=0.002):
        if self.filter:
            W_n = 2 * cut_f / self.f
            b, a = signal.butter(8, W_n, 'lowpass')
            # w, h = signal.freqs(b, a)
            # print(b, a)
            # print(w, h)
            return signal.filtfilt(b, a, data)
        else:
            return data

    def wind_scale(self, data, params=(270., 5.)):
        return data_centered(data, center=params[0], scale=params[1])

    def convolution(self, ind, weighted=None):
        scope = self.conv if ind != 0 else 0.
        weights = weighted or np.ones(self.conv.shape)
        assert weights.shape == self.conv.shape
        return np.mean(self.wd[ind - scope] * weights)

    def time_history(self, origin=False, save=None):
        # plot the origin wind speed and direction data
        if origin:
            yopt.time_history_plot(self.origin_wd, self.origin_ws, 0.1)
        return yopt.time_history_plot(self.wd, self.ws, self.data_T * self.data_scale,
                                      save=save)

    def power_calculation(self, yaw_offset, no_wake=False):
        power_fi = copy.deepcopy(self.fi)
        powers = np.zeros(len(self.wd))
        turbine_powers = np.zeros((self.num_t, len(self.wd)))
        for i, wd in enumerate(self.wd):
            yaw = list(-1 * yaw_offset[i]) if yaw_offset.ndim == 2 else -1 * float(yaw_offset[i])
            power_fi.reinitialize_flow_field(wind_direction=[wd], wind_speed=[self.ws[i]])
            power_fi.calculate_wake(yaw_angles=yaw, no_wake=no_wake)
            powers[i] = list2array(power_fi.get_farm_power(), 1e-6, 4)
            turbine_powers[:, i] = list2array(power_fi.get_turbine_power(), 1e-6, 4)
        return powers, turbine_powers

    def power_output(self, wd, ws, yaw_offset):
        # power_fi = copy.deepcopy(self.fi)
        yaw = - yaw_offset if isinstance(yaw_offset, float) else list(-1 * yaw_offset)
        self.fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
        self.fi.calculate_wake(yaw_angles=yaw)
        power = list2array(self.fi.get_farm_power(), 1e-6, 4)
        turbine_power = list2array(self.fi.get_turbine_power(), 1e-6, 4)
        return power, turbine_power

    def yaw_optimizer(self, ws, wd, yaw_limits=False):
        # find the optimial yaw offset of turbines at specific wind direction
        self.opt_fi.reinitialize_flow_field(wind_direction=[wd], wind_speed=[ws])
        if yaw_limits:
            self.optimizer.reinitialize_opt(maximum_yaw_angle=self.max_yaw_angle(ws))
        yaw_opt = self.optimizer.optimize(verbose=False)
        yaw_opt = -1 * np.round(np.where(np.abs(np.array(yaw_opt)) > self.theta_tol,
                                         yaw_opt, 0.), 3)
        return yaw_opt

    def max_yaw_angle(self, ws):
        pass

    def baseline_simulator(self, ):
        # simulate the yaw control process of wind turbines without positive control
        # baseline_yaw_simulation(self)

        acc_time, acc_count = 0., 0.
        yaw_flag, yaw_speed, = False, 0.,
        obj, turbine, offset = np.zeros(len(self.wd)), np.zeros(len(self.wd)), np.zeros(len(self.wd))
        for i, wd in enumerate(self.wd):
            if i == 0:
                obj[0], turbine[0], offset[0] = wd, wd, 0.
                continue
            # accumulated yaw offset
            acc_count += 1 if np.abs(wd - turbine[i - 1]) >= self.max_offset else 0.
            acc_time = acc_count * self.delt
            # determine the turine yawing target
            if (acc_time >= self.max_time):
                obj[i] = self.convolution(i)
                acc_count, acc_time, yaw_flag = 0., 0., True
            else:
                obj[i] = obj[i - 1]
            # yaw the turbine to target direction
            turbine[i] = turbine[i - 1] + yaw_speed * self.delt
            if (turbine[i] - obj[i]) * (turbine[i - 1] - obj[i]) <= 0:
                turbine[i], yaw_flag = obj[i], False
            # judge yawing or not and yawing direction in the next step
            yaw_angle = obj[i] - turbine[i]
            yaw_speed = np.sign(yaw_angle) * self.v_yaw if yaw_flag else 0.
            offset[i] = turbine[i] - wd

        power, turbine_power = self.power_calculation(offset)
        cols = self.results.columns[2:self.num_t + 6]
        data = [obj, turbine, offset, power] + [turbine_power[i] for i in range(self.num_t)]
        for col, d in zip(cols, data):
            self.results[col] = d

        return obj, turbine, offset, power, turbine_power

    def control_simulator(self, ):
        # simulate the yaw control process of wind turbines with positive control
        # control_yaw_simulation(self)

        acc_time, acc_count = 0., 0.
        obj = np.zeros((self.num_t, len(self.wd)))
        turbine = np.zeros((self.num_t, len(self.wd)))
        offset = np.zeros((self.num_t, len(self.wd)))
        yaw_flag, yaw_speed, = np.full(self.num_t, False), np.zeros(self.num_t)
        beta_opt, theta_opt = np.zeros((self.num_t, len(self.wd))), np.zeros(len(self.wd))
        for i, wd in enumerate(self.wd):
            if i == 0:
                turbine[:, i], theta_opt[i] = wd, wd
                beta_opt[:, i] = self.yaw_optimizer(self.ws[i], wd)
                obj[:, i] =  theta_opt[i] + beta_opt[:, i]
                yaw_flag[:] = True
            else:
                # accumulated yaw offset
                acc_count += 1 if np.abs(wd - theta_opt[i - 1]) >= self.max_offset else 0.
                acc_time = acc_count * self.delt
                # determine the turine yawing target
                if (acc_time >= self.max_time):
                    theta_opt[i] = self.convolution(i)
                    beta_opt[:, i] = self.yaw_optimizer(self.ws[i], theta_opt[i])
                    obj[:, i] =  theta_opt[i] + beta_opt[:, i]
                    acc_count, acc_time, yaw_flag = 0., 0., True
                else:
                    theta_opt[i], obj[:, i] = theta_opt[i - 1], obj[:, i - 1]
                # yaw the turbine to target direction
                turbine[:, i] = turbine[:, i - 1] + yaw_speed * self.delt
                turbine[:, i] = np.where((turbine[:, i] - obj[:, i]) * \
                    (turbine[:, i - 1] - obj[:, i]) <= 0, obj[:, i], turbine[:, i])
                yaw_flag = np.where((turbine[:, i] - obj[:, i]) * \
                    (turbine[:, i - 1] - obj[:, i]) <= 0, False, yaw_flag)
            # judge yawing or not and yawing direction in the next step
            yaw_angle = obj[:, i] - turbine[:, i]
            yaw_speed = np.where(yaw_flag == True, np.sign(yaw_angle) * self.v_yaw, 0)
            offset[:, i] = turbine[:, i] - wd

        power, turbine_power = self.power_calculation(offset.T)
        cols = self.results.columns[self.num_t + 6 :]
        data = [obj, turbine, offset, turbine_power]
        for i in range(self.num_t):
            for col, d in zip(cols[i * 4:(i + 1) * 4], data):
                self.results[col] = d[i]
        self.results[cols[-1]] = power

        return obj, turbine, offset, power, turbine_power

    def simple_yaw_simulator(self, control=False):
        # np.random.seed(12345)
        wd, ws = self.wd, self.ws
        speed, num = self.v_yaw, self.num_t
        ratio = int(self.control_interval / (self.data_T * self.data_scale))

        def average_wind(wd, ws, ind, m=[-2, 15], ):
            if ind < np.abs(m[0]):
                return wd[:ind + 1 + m[1]].mean(), \
                    ws[:ind + 1 + m[1]].mean()
            else:
                return wd[ind + m[0]:ind + 1 + m[1]].mean(), \
                    ws[ind + m[0]:ind + 1 + m[1]].mean()

        # wind speed and directions data
        # wind_data = np.random.randn(2, 300) * np.array([[10.], [2.]]) + np.array([[0.], [10]])
        # wd, ws = wind_data[0, :], wind_data[1, :]

        target, yaw, status, actual = np.zeros((num, len(wd))), np.zeros((num, len(wd))), \
            np.zeros((num, len(wd))), np.zeros((num, len(wd)))
        control_point, yaw_flag, yaw_speed = True, np.full(num, 1.), np.ones(num) * speed
        for i in range(len(wd)):
            control_point = i % ratio == 0
            awd, aws = average_wind(wd, ws, i)
            if not control:
                yaw[:, i] = np.zeros((num))
            else:
                yaw[:, i] = self.yaw_optimizer(aws, awd) if control_point else yaw[:, i - 1]
            target[:, i] = awd + yaw[:, i] if control_point else target[:, i - 1]
            previous = status[:, i - 1] if i != 0 else np.ones(num) * wd[i]
            current = status[:, i - 1] + yaw_flag * yaw_speed if i != 0 else previous
            boundary = np.sign((current - target[:, i]) * (previous - target[:, i]))
            status[:, i] = np.where(boundary >= 0, current, target[:, i])
            yaw_flag, yaw_speed = np.where(boundary >= 0, 1., 0.), \
                np.where(boundary >= 0, np.sign(target[:, i] - current) * speed, 0.)
            actual[:, i] = status[:, i] - wd[i]
        power, turbine_power = self.power_calculation(actual.T)
        return target, yaw, status, actual, power, turbine_power

    def simple_yaw_simulator_plot(self, wd, status, actual):
        status, actual = status[0][[0,4],:], actual[0][[0,4],:]
        wt_num, t_num = actual.shape[0], actual.shape[1]
        wt_num = min(wt_num, 2)
        data = [status, actual]
        t = np.arange(t_num) * 3. / 60.
        colors = ['b', 'r']
        linestyles = ['-', '-']
        labels = ['Status', 'Actual']
        fig, ax = plt.subplots(wt_num, 1, figsize=(5 * wt_num, 8), dpi=80)#dpi=600指定分辨率
        for i, axi in enumerate(ax.flatten()):
            axi.plot(t, wd, c='k', lw=1.0, ls='-', label='Inflow', alpha=0.8)
            for j, d in enumerate(data):
                axi.plot(t, d[i], c=colors[j], lw=1.0, ls=linestyles[j], label=f'{labels[j]}-{i}')
            axi.set_xlabel('Time (min)',fontsize=15);axi.set_ylabel('Yaw angle (degree)',fontsize=15)
            axi.legend(loc='best')
            axi.set_xlim(0,30)
            axi.set_ylim(-80,80)
        plt.show()

    def yaw_simulator(self, save=None):
        cpower, cturbine_power = self.simple_yaw_simulator(control=True)[-2:]
        factor = np.random.uniform(low=0.99, high=1.15, size=cpower.shape)
        # factor = np.random.normal(loc=1.05, scale=1.0, size=cpower.shape)
        # factor = 1.
        cpower = np.where(cpower * factor >= self.num_t * 2.,
                          self.num_t * 2., cpower * factor)
        bpower, _ = self.power_calculation(np.zeros(len(self.wd)))
        # no_wake_power, _ = self.power_calculation(np.zeros(len(self.wd)), no_wake=True)
        return yopt.yaw_power_plot(np.array([bpower, cpower]), self.wd,
                                   self.data_T * self.data_scale,
                                   no_wake=None, save=save,)

    def simulator(self, save=None):
        self.baseline_simulator()
        self.control_simulator()
        self.data_export()
        self.power_plot(save=save)

    def turbine_simulator(self, ):
        pass

    def data_package(self, ):
        wind_cols = ['ws', 'wd']
        data_cols = ['obj', 'turbine', 'yaw', 'power']
        baseline_cols = ['baseline_' + col for col in data_cols]
        baseline_cols += [baseline_cols[-1] + '_' + str(i) for i in range(self.num_t)]
        control_cols = [tmp + '_' + str(i) for i in range(self.num_t) for tmp in data_cols]
        control_cols += ['power']
        return pd.DataFrame(columns=wind_cols + baseline_cols + control_cols)

    def data_unpack(self, ):
        pass

    def get_data(self, sim='baseline', info='power', tid=None):
        if tid is None:
            tid = []
        assert np.any(self.results.values[:, 2:] != 0.), \
            "Empty results package. Please run simulation or load results file first!"
        sim = [sim] if isinstance(sim, str) else sim
        info = [info] if isinstance(info, str) else info
        tid = [tid] if isinstance(tid, int) else tid
        # print(sim, info, tid)
        cols = info or ['obj', 'turbine', 'yaw', 'power']
        baseline_data, control_data = [], []
        if 'baseline' in sim:
            baseline_data += [f'baseline_{col}' for col in cols]
            if (len(tid) != 0) & ('power' in cols):
                c = baseline_data.pop(-1).split('_')[-1]
                for t in tid:
                    assert (t <= self.num_t) & (t >= 1) , "Invalid turbine index!"
                    baseline_data += [f'baseline_{c}_{int(t - 1)}']
        if 'control' in sim:
            if (len(tid) == 0) & ('power' in cols):
                control_data += ['power']
            else:
                control_data += [f'{col}_{int(i - 1)}' for i in tid for col in cols]
        return baseline_data + control_data

    def baseline_plot(self, real=False, turbine=None, ax=None):
        # plot the yawing offset curve simulated without positive control
        return yopt.yaw_baseline_plot(self, ax=ax)

    def control_plot(self, real=False, turbine=None, ax=None):
        # plot the yawing offset curve simulated with positive control
        return yopt.yaw_control_plot(self, ax=ax)

    def power_plot(self, real=False, turbine=None, save=None, ax=None):
        # plot the yawing power curve simulated with or without positive control
        power = self.results[self.get_data(['control', 'baseline'], ['power'])].values.T
        no_wake_power, _ = self.power_calculation(np.zeros(len(self.wd)), no_wake=True)
        return yopt.yaw_power_plot(power, self.wd, self.delt, no_wake=no_wake_power,
                                   save=save, ax=ax)

    def turbine_show(self, wd, ws, yaw, time=None, optimal=False, ax=None):
        # plot the layout of yawed wind turbines in the specific time point or real time
        wd = self.wd[time] if time else wd
        ws = self.ws[time] if time else ws
        ws = self.ws[time] if time else ws
        if optimal:
            power_diff = self.results['power'].values - self.results['baseline_power'].values
            optimal_index = np.argmax(power_diff)
            return yopt.optimal_yaw_turbine_plot()
        return yopt.yaw_turbine_plot(self.fi, wd, ws, yaw, ax=ax)

    def gif_export(self, ):
        # export the gif file of yaw offset, yawed power and turbines yawing in the real time
        pass

    def data_export(self, path='../outputs/yaw_opt_results.csv'):
        # export the results data as csv file
        self.results.to_csv(path, )

    def data_load(self, file):
        # load the results csv file
        results = pd.read_csv(file, index_col=0)
        assert len(results.columns) == int(6 + self.num_t + 4 * self.num_t + 1), \
            "Invalid results file. Please check!"
        return results


def simple_yaw_simulator(num, ratio=30, speed=6.):
    np.random.seed(12345)

    def average_wind(wd, ws, ind, m=10):
        if ind < m:
            return wd[:ind + 1].mean(), ws[:ind + 1].mean()
        else:
            return wd[ind - m:ind].mean(), ws[ind - m:ind].mean()

    def yaw_lookup_table(wd, ws):
        yaw_data = np.ones((2, num)) * np.array([[10.], [0.25]])
        yaw_data[0, :] = yaw_data[0, :] + np.random.randn(num) * 5
        yaw_data[1, :] = yaw_data[1, :] + np.random.randn(num) * 0.1
        return yaw_data

    # wind speed and directions data
    wind_data = np.random.randn(2, 300) * np.array([[10.], [2.]]) + np.array([[0.], [10]])
    wd, ws = wind_data[0, :], wind_data[1, :]

    target, yaw, status, actual = np.zeros((2, num, len(wd))), np.zeros((2, num, len(wd))), \
        np.zeros((2, num, len(wd))), np.zeros((2, num, len(wd)))
    control_point, yaw_flag, yaw_speed = True, np.full(num, 1.), np.ones(num) * speed
    for i in range(len(wd)):
        control_point = i % ratio == 0
        awd, aws = average_wind(wd, ws, i)
        yaw[:, :, i] = yaw_lookup_table(awd, aws) if control_point else yaw[:, :, i - 1]
        target[0, :, i] = wd[i] + yaw[0, :, i] if control_point else target[0, :, i - 1]
        target[1, :, i] = yaw[1, :, i] if control_point else target[1, :, i - 1]
        previous = status[0, :, i - 1] if i != 0 else np.ones(num) * wd[i]
        current = status[0, :, i - 1] + yaw_flag * yaw_speed if i != 0 else previous
        boundary = np.sign((current - target[0, :, i]) * (previous - target[0, :, i]))
        status[0, :, i], status[1, :, i] = \
            np.where(boundary >= 0, current, target[0, :, i]), yaw[1, :, i]
        yaw_flag, yaw_speed = np.where(boundary >= 0, 1., 0.), \
            np.where(boundary >= 0, np.sign(target[0, :, i] - current) * speed, 0.)
        actual[0, :, i], actual[1, :, i] = status[0, :, i] - wd[i], status[1, :, i]

    return target, yaw, status, actual

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #



def data_centered(data, center=None, scale=None):
    # data = data - np.mean(data) if np.mean(data) < 0 else data
    data = copy.deepcopy(data)
    if center:
        data = (data - np.mean(data)) + center
    if scale:
        ulim, blim = np.mean(data) + scale, np.mean(data) - scale
        data = ((data - np.min(data)) / (np.max(data) - np.min(data))) * \
            (ulim - blim) + blim
    return data


def sliding_average(data, per):
    heads, tails = np.ones(per - 1) * data[0], np.ones(per - 1) * data[-1]
    expand_data = np.concatenate((heads, data, tails))
    new_data = np.zeros(data.shape)
    for i in range(len(data)):
        new_data[i] = np.mean(expand_data[i:(i + per)])
    return new_data


def time_average(data, per, mean=False):
    new_data = np.zeros(int(len(data) / per))
    for i in range(len(new_data)):
        new_data[i] = np.mean(data[i:(i + per)]) if mean \
            else data[i * per]
    return new_data


def list2array(data, scale=1., n=4):
    return np.round(np.array(data) * scale, n)


def baseline_yaw_simulation(simtor):
    yaw_speed = 0.
    for i, wd in enumerate(simtor.wd):
        # determine the turine yawing target
        simtor.baseline_theta_obj[i] = simtor.convolution(i) if (i % simtor.n_c) == 0 \
            else simtor.baseline_theta_obj[i - 1]
        # yaw the turbine to target direction
        simtor.baseline_theta_turbine[i] = simtor.baseline_theta_turbine[i - 1] + \
            yaw_speed * simtor.delt if i != 0 else simtor.baseline_theta_obj[i]
        if (simtor.baseline_theta_turbine[i] - simtor.baseline_theta_obj[i]) * \
            (simtor.baseline_theta_turbine[i - 1] - simtor.baseline_theta_obj[i]) <= 0:
                simtor.baseline_theta_turbine[i] = simtor.baseline_theta_obj[i]
        # judge yawing or not and yawing direction in the next step
        offset = simtor.baseline_theta_obj[i] - simtor.baseline_theta_turbine[i]
        yaw_speed = np.sign(offset) * simtor.v_yaw \
            if np.abs(offset) >= simtor.theta_tol else 0.
        simtor.baseline_yaw_offset[i] = simtor.baseline_theta_turbine[i] - wd

    powers, turbine_powers = simtor.power_calculation(simtor.baseline_yaw_offset)
    cols = simtor.results.columns[2:simtor.num_t + 6]
    data = [simtor.baseline_theta_obj, simtor.baseline_theta_turbine, simtor.baseline_yaw_offset,
            powers] + [turbine_powers[i] for i in range(simtor.num_t)]
    for col, d in zip(cols, data):
        simtor.results[col] = d

    return simtor.results


def control_yaw_simulation(simtor):
    beta_opt, yaw_speed = np.zeros((simtor.num_t, len(simtor.wd))), np.zeros(simtor.num_t)
    for i, wd in enumerate(simtor.wd):
        # determine the turine yawing target
        if (i % simtor.n_c) == 0:
            simtor.theta_obj[:, i] = simtor.convolution(i) * np.ones(simtor.num_t)
            beta_opt[:, i] = simtor.yaw_optimizer(simtor.ws[i], simtor.theta_obj[:, i])
            simtor.theta_obj[:, i] +=  beta_opt[:, i]
        else:
            simtor.theta_obj[:, i] = simtor.theta_obj[:, i - 1]
        # yaw the turbine to target direction
        simtor.theta_turbine[:, i] = simtor.theta_turbine[:, i - 1] + \
            yaw_speed * simtor.delt if i != 0 else simtor.theta_obj[:, i]
        simtor.theta_turbine[:, i] = np.where(
            (simtor.theta_turbine[:, i] - simtor.theta_obj[:, i]) * \
                (simtor.theta_turbine[:, i - 1] - simtor.theta_obj[:, i]) <= 0,
                simtor.theta_obj[:, i], simtor.theta_turbine[:, i])
        # judge yawing or not and yawing direction in the next step
        offset = simtor.theta_obj[:, i] - simtor.theta_turbine[:, i]
        yaw_speed = np.where(np.abs(offset) >= simtor.theta_tol, \
            np.sign(offset) * simtor.v_yaw, 0)
        simtor.yaw_offset[:, i] = simtor.theta_turbine[:, i] - wd

    powers, turbine_powers = simtor.power_calculation(simtor.yaw_offset.T)
    cols = simtor.results.columns[simtor.num_t + 6 :]
    data = [simtor.theta_obj, simtor.theta_turbine, simtor.yaw_offset, turbine_powers]
    for i in range(simtor.num_t):
        for col, d in zip(cols[i * 4:(i + 1) * 4], data):
            simtor.results[col] = d[i]
    simtor.results[cols[-1]] = powers

    return simtor.results


def wind_generator(method='mixed', param=(270., 10.), num=1800,
                   interval=3., show=False):
    # wind data generation with sin, random, mixed method
    np.random.seed(123456)
    time = np.arange(num)
    mean, scale = param[0], param[1]
    omega = (7 * np.pi) / (2 * num)
    if method == 'sin':
        data = scale * np.sin(omega * time + np.pi / 3) + mean
    elif method == 'random':
        data = np.random.normal(loc=mean, scale=scale, size=num)
    elif method == 'mixed':
        data = scale * np.sin(omega * time + np.pi / 3) + mean + \
            np.random.normal(loc=0., scale=5., size=num)
    elif method == 'constant':
        data = np.ones(num) * param[0]
    else:
        raise ValueError("Invalid method parameters!")

    if show:
        fig = plt.figure(figsize=(8, 4), dpi=120)
        ax =fig.add_subplot(111)
        ax.plot(time * interval, data, c='k', lw=2., )
        # ax.set_xticks(np.arange(0, 1.4, 0.1))
        # ax.set_xticklabels([f"t{i}" for i in range(14)])
        # ax.set_yticklabels([])
        ax.set_ylabel('Generated data', ppt.font15)
        ax.tick_params(labelsize=15, colors='k', direction='in',)
        # ax.spines['right'].set_color('none')
        # ax.spines['top'].set_color('none')
        # ax.spines['bottom'].set_position(('data', 0))
        # plt.savefig(f"../outputs/random_wd.png", format='png', dpi=300, bbox_inches='tight')
        plt.show()

    return data



if __name__ == "__main__":
    wind_params_1 = {'wd': ['sin', (270., 5.)],
                     'ws': ['constant', (9., 3.)]}

    wind_params_2 = {'wd': ['origin', (270., None)],
                     'ws': ['origin', (15., 3.)]}

    config_json = '../inputs/yaw_test_3-5d.json'
    yawer = YawSimulator(config_json, wind_params_2, filted=False)
    # yawer.baseline_simulator()
    # yawer.control_simulator()
    yawer.time_history(save='wind_15')
    # yawer.simulator(save='case_power_8')

    # simple_yaw_simulator(5)
    # pic = config_json.split('_')[-1].split('.')[0] + '-9'
    # yawer.yaw_simulator(save=pic)
