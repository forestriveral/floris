import time
import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID
from scipy.interpolate import interp1d


class Pid_location(object):
    def __init__(self, exp_val, kp, ki, kd):
        self.KP = kp
        self.KI = ki
        self.KD = kd
        self.exp_val = exp_val
        self.now_val = 0
        self.sum_err = 0
        self.now_err = 0
        self.last_err = 0

    def cmd_pid(self):
        self.last_err = self.now_err
        self.now_err = self.exp_val - self.now_val
        self.sum_err += self.now_err
        self.now_val = self.KP * self.now_err + self.KI * self.sum_err + \
            self.KD * (self.now_err - self.last_err)
        return self.now_val

    def run(self, n=500):
        pid_val = []
        for _ in range(0, n):
            pid_val.append(self.cmd_pid())
        plt.plot(pid_val)
        plt.show()



class Pid_increment():
    def __init__(self, exp_val, kp, ki, kd):
        self.KP = kp
        self.KI = ki
        self.KD = kd
        self.exp_val = exp_val
        self.now_val = 0
        self.now_err = 0
        self.last_err = 0
        self.last_last_err = 0
        self.change_val = 0

    def cmd_pid(self):
        self.last_last_err = self.last_err
        self.last_err = self.now_err
        self.now_err = self.exp_val - self.now_val
        self.change_val = self.KP * (self.now_err - self.last_err) + self.KI * \
            self.now_err + self.KD * (self.now_err - 2 * self.last_err
                                      + self.last_last_err)
        self.now_val += self.change_val
        return self.now_val

    def run(self, n=800):
        pid_val = []
        for i in range(0, 30):
            pid_val.append(self.cmd_pid())
        plt.plot(pid_val)
        plt.show()


class Car:
    """
    被控制系统是一辆汽车
    新的速度=原有速度+加速度-阻力
    """

    def __init__(self):
        self.mass = 100
        self.velocity = 0
        self.accelerated_velocity_arr = []

    def get_current_speed(self, force):
        force_accelerated_velocity = force / self.mass
        wind_accelerated_velocity = - self.velocity * 0.1 / self.mass
        self.velocity = self.velocity + force_accelerated_velocity + wind_accelerated_velocity
        self.accelerated_velocity_arr.append(force_accelerated_velocity + wind_accelerated_velocity)
        return self.velocity


class PIDController:
    """
    输出
    """
    def __init__(self, target_val, kp, ki, kd):
        self.target_val = target_val
        self.controlled_system = Car()

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.out_put_arr = [0]
        self.observed_val_arr = []

        self.now_val = 0
        self.sum_err = 0
        self.now_err = 0
        self.last_err = 0

    def iterate(self):
        self.observed_val_arr.append(self.controlled_system.get_current_speed(self.out_put_arr[-1]))
        self.now_err = self.target_val - self.observed_val_arr[-1]
        # 这一块是严格按照公式来写的
        out_put = self.kp * self.now_err \
            + self.ki * self.sum_err \
            + self.kd * (self.now_err - self.last_err)
        self.out_put_arr.append(out_put)
        self.last_err = self.now_err
        self.sum_err += self.last_err
        return out_put

def car_pid():
    # 对pid进行初始化，目标值是1000 ，Kp=0.1 ，Ki=0.15, Kd=0.1
    controller = PIDController(100, 3, 0.1, 10.)
    # 然后循环100次把数存进数组中去
    for i in range(0, 500):
        controller.iterate()
    # print('controller.out_put_arr,', controller.out_put_arr)
    # print('car.ccelerated_velocity_arr,', controller.controlled_system.accelerated_velocity_arr)
    # print('controller.observed_val_arr,', controller.observed_val_arr)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('force_out_put   (N)', color='red')
    # ax1.set_ylim()
    ax1.plot(controller.out_put_arr, color="red", label='force_out_put')
    ax1.legend(loc=2)

    ax2 = ax1.twinx()
    ax2.plot(controller.observed_val_arr, color='blue', label='car_speed')
    ax2.set_ylabel('car_speed   (Km/h)',color='blue')
    ax2.legend(loc=1)
    plt.title('PID Controller')
    plt.show()


class heater_sys(object):
    def __init__(self):
        self.temp = 25
    def update(self, power, dt):
        if power > 0:
            #加热时房间温度随变量power和时间变量dt 的变化
            self.temp += 2 * power * dt
        #表示房间的热量损失
        self.temp -= 0.5 * dt
        return self.temp


def heater_pid():
	#将创建的模型写进主函数
    heater = heater_sys()
    temp = heater.temp
	#设置PID的三个参数，以及限制输出
    pid = PID(3.0, 1.0, 0.005, setpoint=temp)
    pid.output_limits = (0, None)
	#用于设置时间参数
    start_time = time.time()
    last_time = start_time
	#用于输出结果可视化
    setpoint, y, x = [], [], []
	#设置系统运行时间
    while time.time() - start_time < 10:
       	#设置时间变量dt
        current_time = time.time()
        dt = (current_time - last_time)
        #变量temp在整个系统中作为输出，变量temp与理想值之差作为反馈回路中的输入，通过反馈回路调节变量power的变化。
        power = pid(temp)
        temp = heater.update(power, dt)
        #用于输出结果可视化
        x += [current_time - start_time]
        y += [temp]
        setpoint += [pid.setpoint]
		#用于变量temp赋初值
        if current_time - start_time > 0:
            pid.setpoint = 100

        last_time = current_time
	#输出结果可视化
    plt.plot(x, setpoint, label='target')
    plt.plot(x, y, label='PID')
    plt.xlabel('time')
    plt.ylabel('temperature')
    plt.legend()
    plt.show()


class MyPID(object):
    def __init__(self, P=0.2, I=0.0, D=0.0):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time
        self.clear()

    def clear(self):
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0
        self.windup_guard = 20.0
        self.output = 0.0
        
        self.PTerm_list =[]
        self.ITerm_list =[]
        self.DTerm_list =[]

    def update(self, feedback_value):
        error = self.SetPoint - feedback_value
        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error
        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error  # 比例
            # print(feedback_value, error, self.PTerm)
            self.ITerm += error * delta_time  # 积分
            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard
            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time
            self.last_time = self.current_time
            self.last_error = error
            self.PTerm_list.append(self.PTerm)
            self.ITerm_list.append(self.Ki * self.ITerm)
            self.DTerm_list.append(self.Kd * self.DTerm)
            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        self.Kd = derivative_gain

    def setWindup(self, windup):
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        self.sample_time = sample_time


def test_pid(P=1.5, I=0.0, D=0.0, L=100):
    """Self-test PID class

    .. note::
    ...
    for i in range(1, END):
        pid.update(feedback)
        output = pid.output
        if pid.SetPoint > 0:
            feedback += (output - (1/i))
        if i>9:
            pid.SetPoint = 1
        time.sleep(0.02)
    ---
    """
    pid = MyPID(P, I, D)

    pid.SetPoint=0.0
    pid.setSampleTime(0.01)

    END = L
    feedback = 0

    feedback_list = []
    time_list = []
    setpoint_list = []

    for i in range(1, END):
        pid.update(feedback)
        output = pid.output
        if pid.SetPoint > 0:
            feedback += output  # (output - (1/i))控制系统的函数
        if i > 9:
            pid.SetPoint = 10
        time.sleep(0.01)

        feedback_list.append(feedback)
        setpoint_list.append(pid.SetPoint)
        time_list.append(i)

    time_sm = np.array(time_list)
    time_smooth = np.linspace(time_sm.min(), time_sm.max(), 300)
    interp_func = interp1d(time_list, feedback_list, kind='linear')
    feedback_smooth = interp_func(time_smooth)
    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot()
    ax.plot(time_smooth, feedback_smooth, label='Output', c='r', lw=2., ls='-')
    ax.plot(time_list, setpoint_list, label='Target', c ='k', lw=2., ls='-', alpha=0.7)
    ax.set_xlim((0, L))
    ax.set_ylim((min(feedback_list) - 0.5, max(feedback_list) + 0.5))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('PID (PV)')
    ax.set_title('TEST PID')
    # ax.set_ylim((1 - 0.5, 1 + 0.5))
    ax.legend(loc='upper right')

    ax1 = ax.twinx()
    ax1.plot(time_list, [0] + pid.PTerm_list, label='P Term', c='b', lw=1.5, ls='--')
    ax1.plot(time_list, [0] + pid.ITerm_list, label='I Term', c='g', lw=1.5, ls='--')
    ax1.plot(time_list, [0] + pid.DTerm_list, label='D Term', c='y', lw=1.5, ls='--')
    ax1.set_ylabel('Term values')
    ax1.legend(loc='lower right')

    plt.grid(True)
    plt.show()


def pid_1():
    time_length = 600
    time_sample = 100
    time_interval = float(time_length / time_sample)
    error_coeff = 3
    t = np.linspace(0, time_length, time_sample)
    Slope = 1
    Intercept = 0
    standard_in = 20

    # The system model
    system_model = lambda i: Slope * i + Intercept
    standard_out = system_model(standard_in)
    print("The Standard Output:%d" % standard_out)

    Kp = 0.7  # average
    Ki = 0.  # intergre
    Kd = 0.  # diff

    error_bef = []
    real_out_ajust = []
    real_out_ajust.append(70)
    real_out_ajust.append(75)
    error_bef.append(real_out_ajust[0] - standard_out)
    Out_plt = np.linspace(standard_out, standard_out, time_sample)

    # 标准直接计算公式1：Pout=Kp*e(t) + Ki*Sum[e(t)] + Kd*[e(t) - e(t-1)]
    def PID_Controller_Direct_Mem(standard_out, t):
        # global time_sample, Kp, Ki, Kd, error_bef, real_out_ajust
        if t > time_sample:
            print("Time Out! Quit!")
            return -1
        error_now = real_out_ajust[t] - standard_out
        error_bef.append(error_now)  # 记录了所有的误差
        integrate_res = np.sum(error_bef)
        Diffirent_res = error_now - error_bef[t - 1]
        return Kp * error_now + Ki * integrate_res + Kd * Diffirent_res

    for t_slice in range(1, time_sample - 1):
        Pout = PID_Controller_Direct_Mem(standard_out, t_slice)
        real_out_ajust.append(system_model(Pout))

    plt.figure('PID_Controller_Direct_Mem')
    plt.xlim(0, time_length)
    # plt.ylim(0, 2 * standard_out)
    plt.plot(t, real_out_ajust)
    plt.plot(t, Out_plt)
    plt.show()


if __name__ == "__main__":

    # my_Pid = Pid_location(1000, 0.99, 0., 0.)
    # my_Pid.run()

    # my_Pid = Pid_increment(1000, 0.1, 0.15, 0.1)
    # my_Pid.run()

    # car_pid()

    heater_pid()

    # test_pid(0.5, 2.5, 0.002, L=80)

    # pid_1()
