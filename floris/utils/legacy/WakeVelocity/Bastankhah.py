import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import integrate

from floris.utils.module.tools import power_calc_ops_old as power_ops


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                      MAIN                                    #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class BastankhahWake(object):
    def __init__(self, loc, C_t, D_r, z_hub, I_a=0.077, ytheta=0.,
                 T_m=None, I_w=None, exclusion=True):
        self.ref_loc = loc  # (x_axis, y_axis)
        self.C_thrust = C_t
        self.d_rotor = D_r
        self.z_hub = z_hub
        self.I_a = I_a
        self.epsilon = 0.2 * np.sqrt(
            (1. + np.sqrt(1 - self.C_thrust)) / (2. * np.sqrt(1 - self.C_thrust)))

        self.T_m = T_m
        self.I_wake = None if T_m is None else I_w
        self.k_star = 0.033 if T_m is None else 0.3837 * I_w + 0.003678
        # self.k_star = 0.033 if T_m is None else (0.033 * I_w) / I_a
        self.r_D_ex, self.x_D_ex = self.wake_exclusion if exclusion else (10., 20.)
        self.ytheta = ytheta

    def wake_sigma_Dr(self, x_D):
        return self.k_star * x_D + self.epsilon

    def deficit_constant(self, sigma_Dr):
        flag = self.C_thrust / (8 * sigma_Dr**2)
        if flag >= 1:
            # forcely change A(0D) to 0.99
            flag = 0.9999
            x_D = (sigma_Dr - self.epsilon) / self.k_star
            self.epsilon = np.sqrt(self.C_thrust / (8 * flag))
            sigma_Dr = self.wake_sigma_Dr(x_D)
        A = 1. - np.sqrt(1. - flag)
        B = -0.5 / ((self.d_rotor**2) * (sigma_Dr**2))
        return A, B

    def wake_integrand(self, sigma_Dr, d_spanwise):
        A, B = self.deficit_constant(sigma_Dr)
        return lambda r, t: A * np.exp(
            B * ((r * np.cos(t) + d_spanwise)**2 + (r * np.sin(t))**2)) * r

    def wake_velocity(self, inflow, x, y, z):
        A, B = self.deficit_constant(self.wake_sigma_Dr(x / self.d_rotor))
        v_deficit = A * np.exp(B * ((z - self.z_hub)**2 + y**2))
        return inflow * (1 - v_deficit)

    @staticmethod
    def wake_intersection(d_spanwise, r_wake, down_d_rotor):
        return power_ops.wake_overlap(d_spanwise, r_wake, down_d_rotor)

    @property
    def wake_exclusion(self, m=0.01, n=0.05):
        x_D = np.arange(2, 40, 1)
        A, B = np.vectorize(self.deficit_constant)(self.wake_sigma_Dr(x_D))
        C = np.where(np.log(m / A) > 0., 0., np.log(m / A))
        r_D_ex = np.max(np.sqrt(C / B) / self.d_rotor)
        x_D_ex = (np.sqrt(self.C_thrust / (8 * (1 - (1 - n)**2))) \
            - self.epsilon) / self.k_star
        return r_D_ex, x_D_ex

    def wake_offset(self, ytheta, distance):
        ytheta, distance = ytheta / 360 * 2 * np.pi, distance / self.d_rotor
        theta_func = lambda x_D: np.tan(
            np.cos(ytheta)**2 * np.sin(ytheta) * self.C_thrust * 0.5 * (1 + 0.09 * x_D)**-2)
        offset = integrate.quad(theta_func, 0, distance)[0] * self.d_rotor
        return offset

    def wake_loss(self, down_loc, down_d_rotor):
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches
        assert self.ref_loc[1] >= down_loc[1], "Reference WT must be upstream downstream WT!"
        d_streamwise,  d_spanwise = \
            np.abs(self.ref_loc[1] - down_loc[1]), np.abs(self.ref_loc[0] - down_loc[0])
        if d_streamwise == 0.:
            return 0., 0.
        sigma_Dr = self.wake_sigma_Dr(d_streamwise / self.d_rotor)
        if (d_spanwise / self.d_rotor) < self.r_D_ex or \
            (d_streamwise / self.d_rotor) < self.x_D_ex:
            wake_offset = self.wake_offset(self.ytheta, d_streamwise)
            if self.ref_loc[0] - down_loc[0] == 0:
                d_spanwise = np.abs(wake_offset)
            elif self.ref_loc[0] - down_loc[0] > 0:
                d_spanwise = np.abs(d_spanwise + wake_offset) if wake_offset >= 0 \
                    else np.abs(np.abs(d_spanwise) - np.abs(wake_offset))
            else:
                d_spanwise = np.abs(np.abs(d_spanwise) - np.abs(wake_offset)) \
                    if wake_offset >= 0 else np.abs(d_spanwise + wake_offset)
            # f = lambda r, t: A * np.exp(B * ((r * np.cos(t) + d_spanwise)**2 + (r * np.sin(t))**2)) * r
            integral_velocity, _ = integrate.dblquad(
                self.wake_integrand(sigma_Dr, d_spanwise),
                0, 2 * np.pi, lambda r: 0, lambda r: down_d_rotor / 2)
            integral_velocity = integral_velocity if integral_velocity > 1e-5 else 0.
            intersect_ratio = self.wake_intersection(
                    d_spanwise, 4 * sigma_Dr * self.d_rotor * 0.5, down_d_rotor) \
                        if self.T_m is not None else 0.
            I_add = self.T_m(self.C_thrust, self.I_wake, d_streamwise / self.d_rotor) \
                if self.T_m is not None else 0.
            return integral_velocity / (0.25 * np.pi * down_d_rotor**2), I_add * intersect_ratio
        else:
            return 0., 0.


def BP_data_generator(num=500, test=False):
    # d_rotor: (30, 50)
    # down_d_rotor: (30, 50)
    # C_thrust: (0., 1.)
    # sigma_Dr: (0.15, 3.5)
    # d_spanwise: (0., 20 * sigma_Dr * d_rotor)

    data = np.zeros((num, 6))
    d_rotor = np.random.uniform(30, 50, num)
    down_d_rotor = np.random.uniform(30, 50, num)
    C_thrust = np.random.uniform(0., 1., num)
    # epsilon = 0.2 * np.sqrt((1. + np.sqrt(1 - C_thrust)) / (2. * np.sqrt(1 - C_thrust)))
    # sigma_Dr = np.random.uniform(epsilon, 2. + epsilon, num)
    sigma_Dr = np.random.uniform(0.15, 3.5, num)
    d_spanwise = np.random.uniform(0., 20 * sigma_Dr * d_rotor, num)

    def velocity(d_rotor, down_d_rotor, C_thrust, sigma_Dr, d_spanwise):
        A = 1. - np.sqrt(1. - C_thrust / (8 * sigma_Dr**2)) if C_thrust / (8 * sigma_Dr**2) <= 1. else 1.
        B = - 0.5 / ((d_rotor**2) * (sigma_Dr**2))
        func = lambda r, t: A * np.exp(B * ((r * np.cos(t) + d_spanwise)**2 + (r * np.sin(t))**2)) * r
        integral_velocity, _ = integrate.dblquad(func, 0, 2 * np.pi, lambda r: 0, lambda r: down_d_rotor / 2)
        return integral_velocity

    data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4] = \
        d_rotor, down_d_rotor, C_thrust, sigma_Dr, d_spanwise

    start = time.time()
    if test:
        if test == "vect":
            data[:, -1] = np.vectorize(velocity)(
                data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4])
        if test == "iter":
            for i in range(num):
                data[i, -1] = velocity(
                    data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4])
        end = time.time()
        print(f"{test} | Using time: {end - start}")
    else:
        data[:, -1] = np.vectorize(velocity)(
            data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4])

    return data


class BPWakeGenerator(object):
    def __init__(self, vel, turb, C_thrust, offset, D_rotor, z_hub):
        self.velocity = vel
        self.turbulence = turb
        self.C_thrust = C_thrust
        self.offset = offset
        self.D_rotor = D_rotor
        self.k_star = 0.3837 * self.turbulence + 0.003678
        self.epsilon = 0.2 * np.sqrt(
            (1. + np.sqrt(1 - self.C_thrust)) / (2. * np.sqrt(1 - self.C_thrust)))

    def wake_sigma_Dr(self, x_D):
        return self.k_star * x_D + self.epsilon

    def wake_constant(self, sigma_Dr):
        A = 1. - np.sqrt(1. - self.C_thrust / (8 * sigma_Dr**2)) \
            if self.C_thrust / (8 * sigma_Dr**2) <= 1. else 1.
        B = - 0.5 /  (sigma_Dr**2)
        return A, B

    def wake_offset(self, x_D):
        theta = self.offset / 360 * 2 * np.pi
        theta_func = lambda x_D: np.tan(np.cos(theta)**2 * np.sin(theta) * \
            self.C_thrust * 0.5 * (1 + 0.09 * x_D)**-2)
        offset =  integrate.quad(theta_func, 0, x_D)[0] * self.D_rotor
        return offset

    def wake_mesh(self, xb, yb, zb):
        # xb = x axis boundary; yb = y axis boundary; zb = z axis boundary
        # xb: (min, max, num); yb: (min, max, num); zb: (min, max, num)
        xs = np.linspace(xb[0], xb[1], xb[2])
        zs, ys = np.meshgrid(np.linspace(zb[0], zb[1], zb[2]),
                             np.linspace(yb[0], yb[1], yb[2]))
        return xs, ys, zs

    def wake_section(self, y_D, z_D, A, B):
        deficit = A * np.exp(B * (z_D**2 + y_D**2))
        return self.velocity * (1 - deficit)

    def wake_field(self, xb=(1, 10, 10), yb=(-2, 2, 5), zb=(-1.5, 1.5, 5)):
        xs, ys, zs = self.wake_mesh(xb, yb, zb)
        wake_flow = np.zeros((xb[2], yb[2], zb[2]))

        for i, x_D in enumerate(xs):
            A_i, B_i = self.wake_constant(self.wake_sigma_Dr(x_D))
            d_D = self.wake_offset(x_D) / self.D_rotor
            wake_flow[i, :, :] = \
                np.vectorize(self.wake_section)(ys + d_D, zs, A_i, B_i)[:,::-1].T
        return wake_flow

    @property
    def wake_exclusion(self):
        m, n = 0.01, 0.05
        x_D = np.arange(0, 40, 1)
        A, B = np.vectorize(self.wake_constant)(self.wake_sigma_Dr(x_D))
        print(self.C_thrust, self.wake_sigma_Dr(x_D), A, B)
        rs_D_ex = np.sqrt(np.log(m / A) / B)
        x_D_ex = (np.sqrt(self.C_thrust / (8 * (1 - (1 - n)**2))) - self.epsilon) / self.k_star
        # plt.figure(figsize=(12, 8))
        # plt.plot(x_D, A, 'k-', lw=2.)
        # plt.savefig("output/wake_A.png", format='png',
        #             dpi=300, bbox_inches='tight')
        return np.max(rs_D_ex), x_D_ex


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                            MISCELLANEOUS                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def bast_wake(x_d0, r, k, c):
    beta = (1. + np.sqrt(1 - c)) / (2. * np.sqrt(1 - c))
    e = 0.2 * np.sqrt(beta)
    print(e)
    o_d0 = k * x_d0 + e
    A = 1. - np.sqrt(1. - (c / (8 * o_d0**2)))
    v_deficit = A * np.exp(- (r**2) / (2 * o_d0**2))
    return v_deficit


def wake_plot(wake, x_d0, k, c, r=None, z0=None):
    if not isinstance(x_d0, list):
        x_d0 = [x_d0]
    if z0:
        k = k_decay_constant(np.array(z0), zh=70)
        print(k)
    elif not isinstance(k, list) or len(k) == 1:
        k = list(range(len(x_d0)))
    if not r:
        r = np.arange(-2, 2, 0.01)
    # print(k)
    plt.figure(num=1, figsize=(15, 8), dpi=100)
    colors = list(mcolors.TABLEAU_COLORS.keys())

    for i, (ki, xi) in enumerate(zip(k, x_d0)):
        # print(x_d0)
        y = wake(xi, r, ki, c)
        plt.plot(r, y, color=colors[i], linewidth=1.0,
                 linestyle="-", label="{}_{}D".format(ki, xi))

    plt.legend()
    plt.show()


def wake_expansion_plot(k, c):
    beta = (1 + np.sqrt(1 - c)) / 2 * np.sqrt(1 - c)
    e = 0.2 * np.sqrt(beta)
    x_d0 = np.arange(0, 15, 1)

    if not isinstance(k, list):
        k = [k]
    plt.figure(num=1, figsize=(15, 8), dpi=100)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i, ki in enumerate(k):
        # print(x_d0)
        o_d0 = ki * x_d0 + e
        plt.plot(x_d0, o_d0, color=colors[i],
                 linewidth=1.0, linestyle="-",
                 label="{}".format(ki))
    plt.legend()
    plt.show()


def k_decay_constant(z0, zh=70):
    epsilon = 1e-10
    return 0.5 / np.log((zh / z0) + epsilon)


def k_plot():
    plt.figure(num=1, figsize=(15, 8), dpi=100)
    z0 = np.arange(0, 1, 0.05)
    k = k_decay_constant(z0)
    plt.plot(z0, k)
    plt.show()




if __name__ == "__main__":
    # k_w = 0.075
    # k_w = [0.04, 0.06, 0.075]
    # # k_w = [0.075, 0.075, 0.075]
    # # z_0 = [5e-1, 5e-2, 5e-3, 5e-5]
    # z_0 = [5e-1, 5e-1, 5e-1, 5e-1]
    # C_t = 0.8
    # D = 80
    # # d = 3
    # d = [3., 3., 3., 3.]
    # d = [3., 5., 7., 9.]
    # # d = [5, 5, 5]
    # # d = [7, 7, 7]
    # # d = [3, 5, 7]

    # wake_plot(bast_wake, d, k_w, C_t, z0=z_0)
    # wake_expansion_plot([0.04, 0.06, 0.075], C_t)
    # k_plot()

    m = 10.
    n = 1.5
    I_w = 0.12
    
    # test = BastankhahWake(np.array([0., 80. * m]), 8, 0.8, 80, 70,
    #                       I_w=I_w)
    # # v_w_1 = test.wake_velocity(500., 0., 70.)
    # # print(v_w_1)
    # std = test.wake_sigma_Dr(80. * m) * 80
    # print(std)
    # print(test.wake_loss(np.array([std * n, 0.]), 80))
    
    # data = BP_data_generator(500)
    # print(data[:10, :])
    
    # BP_data_generator(5000, test="vect")
    # BP_data_generator(5000, test="iter")
    
    wake = BPWakeGenerator(8, 0.077, 0.75, 30, 80, 70)
    # wake_data = wake.wake_field()
    # print(wake_data[5, :, :])
    # print(wake.wake_exclusion)
    
