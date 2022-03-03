import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                   MAIN                                       #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ADMSimulator(object):
    def __init__(self, tsr=None, iteration='default'):
        # air properties
        self.rho = 1.225  # kg/m3
        # turbine properties
        self.blade_num = 3  # blade number
        self.R = 40.  # m
        self.A_rotor = np.pi * self.R ** 2
        self.mech_eff = 1.0
        self.measured_power, self.measured_thrust = \
            self.power_data('../inputs/turbines/Vesta_2MW.json')
        self.chord = self.airfoil_data('../inputs/turbines/chord.txt')  # blade chord length
        self.twist = self.airfoil_data('../inputs/turbines/twist.txt')  # blade twist angle (degree)
        self.C_l = self.airfoil_data('../inputs/turbines/lift.txt')  # blade lift coefficient
        self.C_d = self.airfoil_data('../inputs/turbines/drag.txt')  # blade drag coefficient

        # turbine operation conditions
        self.v_0 = 8.  # m/s
        self.tsr = 8.89  if not tsr else tsr# tip speed ratio
        self.omega = (self.tsr * self.v_0 * 60) / (2 * np.pi * self.R)  # rotation speed (RPM)
        self.theta_pt = 0.  # blade pitch angle (degree)

        self.conv_tol = 1.0e-3
        self.blade_points = np.linspace(0.1, 1, 20)
        self.iter_history, self.induction = self.induction_iter() if iteration == 'default' \
            else self.induction_iter_from_velocity()
        self.a_axial = interp1d(self.blade_points, self.induction[0],
                                kind='cubic', fill_value="extrapolate")
        self.a_tange = interp1d(self.blade_points, self.induction[1],
                                kind='cubic', fill_value="extrapolate")
        self.tsr_r = lambda r: r * self.tsr
        self.v_ref = lambda r: self.v_0 * np.sqrt((1 - self.a_axial(r))**2 + \
            (self.tsr_r(r) * (1 + self.a_tange(r)))**2)
        self.theta_phi = lambda r: np.rad2deg(np.arctan(((1 - self.a_axial(r))) / \
            (self.tsr_r(r) * (1 + self.a_tange(r)))))
        self.theta_attack = lambda r: self.theta_phi(r) - self.theta_pt - self.twist(r)

        self.C_X = lambda r: self.C_l(self.theta_attack(r)) * np.cos(np.deg2rad(self.theta_phi(r))) + \
            self.C_d(self.theta_attack(r)) * np.sin(np.deg2rad(self.theta_phi(r)))
        self.C_Y = lambda r: self.C_l(self.theta_attack(r)) * np.sin(np.deg2rad(self.theta_phi(r))) - \
            self.C_d(self.theta_attack(r)) * np.cos(np.deg2rad(self.theta_phi(r)))

        self.thrust = self.thrust_calc()
        self.C_t = self.thrust / (0.5 * self.rho * self.v_0**2 * self.A_rotor)
        # print(0.5 * self.rho * self.v_0**2 * self.A_rotor)
        self.torque = self.torque_calc()
        self.rotor_power = self.torque * self.omega / (9550. * 1000)  # power unit is MW
        self.power = self.rotor_power * self.mech_eff

    def airfoil_data(self, path):
        data = np.loadtxt(path) if isinstance(path, str) else path
        return interp1d(data[:, 0], data[:, 1], kind='cubic', fill_value="extrapolate")

    def power_data(self, path):
        with open(path) as jsonfile:
            data = json.load(jsonfile)
        table = data["turbine"]["properties"]["power_thrust_table"]
        return interp1d(table["wind_speed"], table["power"], kind='linear'), \
            interp1d(table["wind_speed"], table["thrust"], kind='linear')

    def induction_iter(self, num=100):
        recorder = [[], []]
        induction = [[], []]
        for r in (self.blade_points * self.R):
            count_tol = 0
            a_axials, a_tanges = [0., ], [0., ]
            theta_phi, a_axial, a_tange = 0., 0., 0.
            for i in range(num):
                theta_phi = np.arctan((1 - a_axial) / ((self.tsr * r / self.R) * (1 + a_tange)))
                chord_solidity = (self.blade_num * self.chord(r / self.R)) / (2 * np.pi * r)
                theta_attack = np.rad2deg(theta_phi) - self.theta_pt - self.twist(r / self.R)
                C_x = self.C_l(theta_attack) * np.cos(theta_phi) + self.C_d(theta_attack) * np.sin(theta_phi)
                C_y = self.C_l(theta_attack) * np.sin(theta_phi) - self.C_d(theta_attack) * np.cos(theta_phi)
                g_x = chord_solidity * C_x / (4 * np.sin(theta_phi)**2)
                g_y = chord_solidity * C_y / (4 * np.sin(theta_phi) * np.cos(theta_phi))
                a_axial_pre, a_tange_pre = a_axial, a_tange
                a_axial, a_tange = g_x / (1 + g_x), g_y / (1 - g_y)
                a_axials.append(a_axial), a_tanges.append(a_tange)
                errors = np.abs(np.array([a_axial - a_axial_pre, a_tange - a_tange_pre]))
                if np.all(errors < self.conv_tol):
                    count_tol += 1
                if (count_tol > 5) | (i >= num - 1):
                    induction[0].append(a_axial), induction[1].append(a_tange)
                    break
            recorder[0].append(a_axials), recorder[1].append(a_tanges)
        return recorder, induction

    def induction_iter_from_velocity(self, num=100):
        recorder = [[], []]
        induction = [[], []]
        for r in (self.blade_points * self.R):
            count_tol = 0
            a_axials, a_tanges = [0, ], [0, ]
            a_axial, a_tange = 0., 0.
            theta_phi, v_axial, v_tange_r = 0., self.v_0, self.v_0 * self.tsr / self.R
            for i in range(num):
                theta_phi = np.arctan(v_axial / (v_tange_r * r ))
                chord_solidity = (self.blade_num * self.chord(r / self.R)) / (2 * np.pi * r)
                theta_attack = np.rad2deg(theta_phi) - self.theta_pt - self.twist(r / self.R)
                C_x = self.C_l(theta_attack) * np.cos(theta_phi) + self.C_d(theta_attack) * np.sin(theta_phi)
                C_y = self.C_l(theta_attack) * np.sin(theta_phi) - self.C_d(theta_attack) * np.cos(theta_phi)
                g_x = chord_solidity * C_x / (4 * np.sin(theta_phi)**2)
                g_y = chord_solidity * C_y / (4 * np.sin(theta_phi) * np.cos(theta_phi))
                v_axial_pre, v_tange_r_pre = v_axial, v_tange_r
                a_axial, a_tange = g_x / (1 + g_x), g_y / (1 - g_y)
                a_axials.append(a_axial), a_tanges.append(a_tange)
                v_axial, v_tange_r = self.v_0 * (1 - a_axial), \
                    self.v_0 * self.tsr / self.R * (1 + a_tange)
                errors = np.abs(np.array([v_axial - v_axial_pre, v_tange_r - v_tange_r_pre]))
                if np.all(errors < self.conv_tol):
                    count_tol += 1
                if (count_tol > 5) | (i >= num - 1):
                    induction[0].append(a_axial), induction[1].append(a_tange)
                    break
            recorder[0].append(a_axials), recorder[1].append(a_tanges)
        return recorder, induction

    def induction_plot(self, ):
        a_axials, a_tanges = self.iter_history[0], self.iter_history[1]
        fig = plt.figure(figsize=(12, 5), dpi=100)
        colors = plt.get_cmap('bwr')(np.linspace(0, 1, len(self.blade_points)))
        ax1 = fig.add_subplot(131)
        for i in range(len(self.blade_points)):
            ax1.plot(np.arange(len(a_axials[i])), a_axials[i], c=colors[i], ls='-',
                    lw=1.5, label=f'r={self.blade_points[i]:.2f}')
        ax1.legend(loc='best')

        ax2 = fig.add_subplot(132)
        for i in range(len(self.blade_points)):
            ax2.plot(np.arange(len(a_tanges[i])), a_tanges[i], c=colors[i], ls='-',
                    lw=1.5, label=f'r={self.blade_points[i]:.2f}')
        ax2.legend(loc='best')

        ax3 = fig.add_subplot(133)
        ax3.plot(self.blade_points, self.a_axial(self.blade_points), c='k', ls='-', lw=2., label=f'Axial')
        ax3.plot(self.blade_points, self.a_tange(self.blade_points), c='b', ls='-', lw=2., label=f'Tangential')
        ax3.set_xlim([0, 1.1])
        ax3.legend(loc='best')

        # plt.savefig('../outputs/induction.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()

    def thrust_calc(self, ):
        thrust_func = lambda r: 0. if (r < 0.15) | (r > 1.) else \
            self.v_ref(r)**2 * self.chord(r) * self.C_X(r)
        thrust, _ = quad(thrust_func, 0., 1., limit=100)
        return 0.5 * self.rho * self.blade_num * self.R * thrust

    def torque_calc(self, ):
        torque_func = lambda r: 0. if (r < 0.15) | (r > 1.) else \
            self.v_ref(r)**2 * self.chord(r) * self.C_Y(r) * r
        torque, _ = quad(torque_func, 0., 1., limit=100)
        return 0.5 * self.rho * self.blade_num * self.R**2 * torque


def attack_angle_comparsion():
    tsr_list = np.arange(2, 13, 2)
    blade_points = np.linspace(0.1, 1, 20)
    fig = plt.figure(figsize=(6, 5), dpi=120)
    colors = plt.get_cmap('bwr')(np.linspace(0, 1, len(tsr_list)))
    ax1 = fig.add_subplot(111)
    for i, tsr in enumerate(tsr_list):
        attack = ADMSimulator(tsr).theta_attack(blade_points)
        ax1.plot(blade_points, attack, c=colors[i], ls='-',
                 lw=1.5, label=f'r={tsr:.2f}')
    ax1.legend(loc='best')

    plt.show()


def iteration_comparsion():
    method = ['default', 'velocity']
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot()
    colors = ['k', 'b']
    markers = ['o', 'x']
    for i, m in enumerate(method):
        adm = ADMSimulator(iteration=m)
        ax.plot(adm.blade_points, adm.a_axial(adm.blade_points), markersize=4,
                marker=markers[i], c=colors[i], ls='-',lw=2., label=f'Axial({m})')
        ax.plot(adm.blade_points, adm.a_tange(adm.blade_points), markersize=4,
                marker=markers[i], c=colors[i], ls='--', lw=2., label=f'Tangential({m})')
    ax.set_xlim([0, 1.1])
    ax.set_xlabel('r/R')
    ax.legend(loc='best')
    plt.show()


def lift_drag_fitting():
    Cd = [0.036, 0.039, 0.07, 0.18, 0.268, 0.344, 0.426, 0.536, 0.681, 0.772, 0.829,
          0.886, 0.938, 1, 1.069, 1.108, 1.145, 1.167, 1.183, 1.18, 1.166, 1.148, 1.112,
          1.044, 1.003, 0.965, 0.902, 0.825, 0.702, 0.602, 0.508, 0.429, 0.28, 0.191, 0.012,
          0.011, 0.011, 0.01, 0.01, 0.009, 0.009, 0.008, 0.008, 0.008, 0.007, 0.007, 0.008,
          0.008, 0.009, 0.011, 0.012, 0.013, 0.015, 0.019, 0.024, 0.03, 0.039, 0.049, 0.062,
          0.077, 0.094, 0.288, 0.444, 0.578, 0.721, 0.811, 0.874, 1.022, 1.095, 1.144, 1.179,
          1.235, 1.268, 1.272, 1.265, 1.26, 1.254, 1.205, 1.127, 1.094, 0.978, 0.872, 0.805,
          0.732, 0.633, 0.616, 0.466, 0.308, 0.271, 0.195, 0.073, 0.028, 0.036]
    Cl = [-0.088, 0.083, 0.701, 1.081, 0.661, 0.477, 0.612, 0.697, 0.757, 0.716, 0.643,
          0.572, 0.537, 0.466, 0.361, 0.287, 0.179, 0.094, 0.022, -0.099, -0.192, -0.261,
          -0.332, -0.437, -0.538, -0.597, -0.656, -0.673, -0.68, -0.698, -0.655, -0.535,
          -0.386, -0.354, -0.463, -0.6, -0.504, -0.382, -0.261, -0.139, -0.017, 0.104,
          0.226, 0.347, 0.449, 0.574, 0.694, 0.825, 0.962, 1.073, 1.142, 1.211, 1.288,
          1.355, 1.418, 1.396, 1.36, 1.319, 1.301, 1.25, 1.209, 0.806, 0.825, 0.96, 0.955,
          0.926, 0.865, 0.697, 0.626, 0.495, 0.401, 0.303, 0.199, 0.099, -0.014, -0.112,
          -0.214, -0.317, -0.427, -0.539, -0.629, -0.695, -0.796, -0.77, -0.729, -0.721,
          -0.674, -0.597, -0.843, -1.324, -0.88, -0.475, -0.088]
    alpha = [-180, -177.79, -172.18, -168.33, -165.1, -159.59, -154.91, -150.61, -145.49,
             -141.27, -134.54, -127.58, -124.17, -120.56, -115.24, -109.23, -104.33, -100.65,
             -96.72, -88.29, -82.73, -78.79, -74.68, -68.72, -62.9, -58.21, -53.67, -50.77,
             -44.08, -38.09, -32.99, -29.64, -24.44, -19, -12.78, -9.56, -8, -7,- 6, -5, -4,
             -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 21.61,
             26.07, 30.47, 36.24, 43.09, 47.56, 55.54, 60.67, 66.93, 71.07, 75.5, 80.75,
             86.9, 91.87, 95.79, 99.84, 104.93, 110.44, 115.85, 124.05, 129.73, 136.34,
             140.53, 144.46, 145.07, 150.54, 155.32, 160.62, 163.39, 168.46, 175.01, 180]

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(alpha, Cl, c='k', label='lift')
    ax.plot(alpha, Cd, c='b', label='drag')
    ax.legend()

    plt.show()


if __name__ == "__main__":
    # adm = ADMSimulator()
    # adm.induction_plot()

    # print(f"Torque: {adm.torque:.2f} (N.m)")
    # print(f"C_T: {adm.C_t:.4f}  Measured C_T: {adm.measured_thrust(adm.v_0):.4f}")
    # print(f"Power: {adm.power:.4f} (MW)  Measured Power: {adm.measured_power(adm.v_0):.4f}")

    # attack_angle_comparsion()
    # lift_drag_fitting()
    iteration_comparsion()
