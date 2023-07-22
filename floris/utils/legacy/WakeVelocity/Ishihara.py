import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def Ishihara_wake(x_D, y, z, C, I, D, H, verbose=False):
    r_D = np.sqrt((D ** 2) * y ** 2 + (H ** 2) * (z - 1) ** 2) / D

    k = 0.11 * C**1.07 * I**0.20
    ep = 0.23 * C**-0.25 * I**0.17

    a = 0.93 * C**-0.75 * I**0.17
    b = 0.42 * C**0.6 * I**0.2
    c = 0.15 * C**-0.25 * I**-0.7

    o_D = k * x_D + ep
    A = 1. / (a + b * x_D + c * (1 + x_D)**-2)**2

    if verbose:
        print("k: ", k)
        print("ep: ", ep)
        print("a: ", a)
        print("b: ", b)
        print("c: ", c)
        print("o_D: ", o_D)
        print("A: ", A)

    v_deficit = A * np.exp(- (r_D**2) / (2 * o_D**2))
    print(v_deficit)
    return v_deficit


def Ishihara_turbulence(x_D, y, z, C, I, D, H, verbose=False):
    r_D = np.sqrt((D ** 2) * y ** 2 + (H ** 2) * (z - 1) ** 2) / D

    k = 0.11 * C**1.07 * I**0.20
    ep = 0.23 * C**-0.25 * I**0.17

    d = 2.3 * C**-1.2
    e = 1.0 * I**0.1
    f = 0.7 * C**-3.2 * I**-0.45

    def k_1(r_D):
        return np.where(r_D <= 0.5, np.cos(np.pi / 2 * (r_D - 0.5)) ** 2, 1.)

        # r_D = np.where(r_D > 0.5, 0.5, r_D)
        # return np.cos(np.pi / 2 * (r_D - 0.5)) ** 2

        # if r_D <= 0.5:
        #     return np.cos(np.pi / 2 * (r_D - 0.5))**2
        # else:
        #     return 1.

    def k_2(r_D):
        return np.where(r_D <= 0.5, np.cos(np.pi / 2 * (r_D + 0.5)) ** 2, 0.)

        # r_D = np.where(r_D > 0.5, 0.5, r_D)
        # return np.cos(np.pi / 2 * (r_D + 0.5)) ** 2

        # if r_D <= 0.5:
        #     return np.cos(np.pi / 2 * (r_D + 0.5))**2
        # else:
        #     return 0.

    def zz(z, H):
        return np.where(z < H, I * np.sin(np.pi * (1 - z))**2, 0.)

        # if z < H:
        #     return I * np.sin(np.pi * (H - z) / H)**2
        # else:
        #     return 0.

    o_D = k * x_D + ep
    B = 1. / (d + e * x_D + f * (1 + x_D)**-2)
    if verbose:
        print("k: ", k)
        print("ep: ", ep)
        print("d: ", d)
        print("e: ", e)
        print("f: ", f)
        print("k1: ", k_1(r_D))
        print("k2: ", k_2(r_D))
        print("z: ", z)
        print("o_D: ", o_D)
        print("B: ", B)

    added_I = B * (k_1(r_D) * np.exp(- ((r_D - 1 / 2)**2) / (2 * o_D**2)) +
                   k_2(r_D) * np.exp(- ((r_D + 1 / 2)**2) / (2 * o_D**2))) - zz(z, H)
    return added_I


def wake_plot(x_D, C, I, D, H, y=None, z=None, verbose=False):
    fig = plt.figure(num=1, dpi=100)
    if (y is None) and (z is None):
        ax = Axes3D(fig)
        y = np.arange(-1, 1, 0.05)
        z = np.arange(0, 2, 0.05)
        y, z = np.meshgrid(y, z)
        ax.plot_surface(y, z, Ishihara_wake(x_D, y, z, C, I, D, H, verbose=verbose),
                        rstride=1, cstride=1, cmap=plt.cm.hot)
        ax.view_init(elev=60, azim=125)
    else:
        colors = list(mcolors.TABLEAU_COLORS.keys())
        if y is not None:
            z = np.arange(0, 3, 0.05)
            plt.plot(z, Ishihara_wake(x_D, y, z, C, I, D, H), color=colors[0], linewidth=1.0,
                     linestyle="-", label="{}D".format(x_D))
        if z is not None:
            y = np.arange(-1, 1, 0.05)
            plt.plot(y, Ishihara_wake(x_D, y, z, C, I, D, H), color=colors[0], linewidth=1.0,
                     linestyle="-", label="{}D".format(x_D))
        # ax = plt.gca()
        # ax.xaxis.set_ticks_position('left')
        # ax.yaxis.set_ticks_position('bottom')
        plt.legend()

    # plt.show()


def turbulence_plot(x_D, C, I, D, H, y=None, z=None, verbose=False):
    fig = plt.figure(num=1, dpi=100)
    if (y is None) and (z is None):
        ax = Axes3D(fig)
        y = np.arange(-1, 1, 0.05)
        z = np.arange(0, 2, 0.05)
        y, z = np.meshgrid(y, z)
        # ax.plot_surface(y, z, ishihara_turbulence(x_D, y, z, C, I, D, H, verbose=verbose),
        #                 rstride=1, cstride=1, cmap=plt.cm.hot)
        ax.view_init(elev=60, azim=125)
        output_txt(Ishihara_turbulence(x_D, y, z, C, I, D, H, verbose=verbose))

    else:
        colors = list(mcolors.TABLEAU_COLORS.keys())
        if y is not None:
            z = np.arange(0, 3, 0.05)
            ax = plt.gca()
            plt.plot(z, Ishihara_turbulence(x_D, y, z, C, I, D, H), color=colors[0], linewidth=1.0,
                     linestyle="-", label="{}D".format(x_D))
            # ax.xaxis.set_ticks_position('left')
            # ax.yaxis.set_ticks_position('bottom')
            plt.legend()

    # plt.show()


def output_txt(input):
    df = pd.DataFrame(input)
    df.to_csv('./matrix.txt')
    # print(df)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def _Ishihara_wake(x, y, z, v_inflow, D_r, C_t, I_a, z_hub):
    k_star = 0.11 * (C_t**1.07) * (I_a**0.20)
    epsilon = 0.23 * (C_t**-0.25) * (I_a**0.17)
    a = 0.93 * (C_t**-0.75) * (I_a**0.17)
    b = 0.42 * (C_t**0.6) * (I_a**0.2)
    c = 0.15 * (C_t**-0.25) * (I_a**-0.7)
    d = 2.3 * (C_t**-1.2)
    e = 1.0 * (I_a**0.1)
    f = 0.7 * (C_t**-3.2) * (I_a**-0.45)

    r = np.sqrt(((z - z_hub)**2) + (y**2))
    sigma_D_r = k_star * (x / D_r) + epsilon

    A = 1. / ((a + b * (x / D_r) + c * ((1 + (x / D_r))**-2))**2)
    v_deficit = A * np.exp(- ((r / D_r)**2) / (2 * (sigma_D_r**2)))
    v_wake = v_inflow * (1 - v_deficit)

    def k_1(r, D):
        return 1. if (r / D) > 0.5 else np.cos((np.pi / 2) * ((r / D) - 0.5))**2

    def k_2(r, D):
        return 0. if (r / D) > 0.5 else np.cos((np.pi / 2) * ((r / D) + 0.5))**2

    def delta(z, H, I):
        return 0. if z >= H else I * np.sin(np.pi * (1 - (z / H)))**2

    B = 1. / (d + e * (x / D_r) + f * ((1 + (x / D_r))**-2))
    I_add = B * (k_1(r, D_r) * np.exp(- (((r / D_r) - 0.5)**2) / (2 * sigma_D_r**2)) +
                 k_2(r, D_r) * np.exp(- (((r / D_r) + 0.5)**2) / (2 * sigma_D_r**2))) - delta(z, z_hub, I_a)

    return sigma_D_r, v_deficit, v_wake, I_add


class IshiharaWake(object):
    def __init__(self, loc, inflow, C_t, D_r, z_hub, T_m=None, I_w=None,
                 I_a=0.077):
        self.ref_loc = loc  # (x_axis, y_axis)
        self.v_inflow = inflow
        self.C_thrust = C_t
        self.d_rotor = D_r
        self.z_hub = z_hub
        self.I_a = I_a if (I_w is None) or (T_m is None) else I_w
        self.T_m = T_m

        self.k_star = 0.11 * (self.C_thrust**1.07) * (self.I_a**0.20)
        self.epsilon = 0.23 * (self.C_thrust**-0.25) * (self.I_a**0.17)
        self.a = 0.93 * (self.C_thrust**-0.75) * (self.I_a**0.17)
        self.b = 0.42 * (self.C_thrust**0.6) * (self.I_a**0.2)
        self.c = 0.15 * (self.C_thrust**-0.25) * (self.I_a**-0.7)

        self.d = 2.3 * (self.C_thrust**-1.2)
        self.e = 1.0 * (self.I_a**0.1)
        self.f = 0.7 * (self.C_thrust**-3.2) * (self.I_a**-0.45)

    def k_1(self, r):
        if (r / self.d_rotor) > 0.5:
            return 1.
        else:
            return np.cos((np.pi / 2) * ((r / self.d_rotor) - 0.5))**2

    def k_2(self, r):
        if (r / self.d_rotor) > 0.5:
            return 0.
        else:
            return np.cos((np.pi / 2) * ((r / self.d_rotor) + 0.5))**2

    def delta(self, z):
        return 0. if z >= 0 else self.I_a * np.sin(np.pi * (1 - (z / self.z_hub)))**2

    def wake_sigma_Dr(self, x):
        return self.k_star * (x / self.d_rotor) + self.epsilon

    def deficit_constant(self, x):
        a = 1. / ((self.a + self.b * (x / self.d_rotor) +
                   self.c * ((1 + (x / self.d_rotor))**-2))**2)
        b = -0.5 / (self.wake_sigma_Dr(x) * self.d_rotor)**2
        return a, b

    def wake_velocity_integrand(self, d_streamwise, d_spanwise):
        A, B = self.deficit_constant(d_streamwise)
        return lambda r, t: A * np.exp(B * ((r * np.cos(t) + d_spanwise)**2 + (r * np.sin(t))**2)) * r

    def wake_velocity(self, x, y, z):
        A, B = self.deficit_constant(x)
        v_deficit = A * np.exp(B * (((z - self.z_hub)**2) + (y**2)))
        return v_deficit

    def turbulence_constant(self, x):
        a = 1. / (self.d + self.e * (x / self.d_rotor) +
                  self.f * ((1 + (x / self.d_rotor))**-2))
        b = self.deficit_constant(x)[1]
        return a, b

    def wake_turbulence_integrand(self, d_streamwise):
        A, B = self.turbulence_constant(d_streamwise)
        def turbulence(y, z):
            r = np.sqrt((z**2) + (y**2))
            return A * (self.k_1(r) * np.exp(B * ((r - 0.5 * self.d_rotor)**2)) +
                        self.k_2(r) * np.exp(B * ((r + 0.5 * self.d_rotor)**2))) - \
                            self.delta(z)
        def integral_limit(y, r_rotor, d_offset):
                assert (y >= d_offset - r_rotor ) and (y <= d_offset + r_rotor)
                return np.sqrt(r_rotor**2 - (y - d_offset)**2)
        return turbulence, integral_limit

    def wake_turbulence(self, x, y, z):
        A, B = self.turbulence_constant(x)
        r = np.sqrt(((z - self.z_hub)**2) + (y**2))
        added_turbulence = A * (self.k_1(r) * np.exp(B * ((r - 0.5 * self.d_rotor)**2)) +
                                self.k_2(r) * np.exp(B * ((r + 0.5 * self.d_rotor)**2))) - \
                                    self.delta(z)
        return added_turbulence

    def wake_loss(self, down_loc, down_d_rotor, down_z_hub=None, eq=None):
        assert self.ref_loc[1] >= down_loc[1], "Reference WT must be upstream downstream WT!"
        d_streamwise,  d_spanwise = \
            np.abs(self.ref_loc[1] - down_loc[1]), np.abs(self.ref_loc[0] - down_loc[0])
        sigma_Dr = self.wake_sigma_Dr(d_streamwise)
        m, n, = 20., 3.  # application scope of the model and control the calculation
        if d_spanwise > ( 0.5 * self.d_rotor + n * sigma_Dr * self.d_rotor) or d_streamwise > m * self.d_rotor:
            return 0., 0.
        elif d_streamwise == 0:
            return 0., 0.
        else:
            integral_velocity, _ = integrate.dblquad(
                self.wake_velocity_integrand(d_streamwise, d_spanwise),
                0, 2 * np.pi, lambda r: 0, lambda r: down_d_rotor / 2)
            integral_turbulence = 0.
            if self.T_m in ["Ishihara",]:
                turbulence, limit = self.wake_turbulence_integrand(d_streamwise)
                integral_turbulence, _ = \
                    integrate.dblquad(turbulence, d_spanwise - (down_d_rotor / 2),
                                      d_spanwise + (down_d_rotor / 2),
                                      lambda y : - limit(y, down_d_rotor / 2, d_spanwise),
                                      lambda y : limit(y, down_d_rotor / 2, d_spanwise))
                integral_turbulence = integral_turbulence / (0.25 * np.pi * down_d_rotor**2)
            else:
                integral_turbulence = self.wake_turbulence(d_streamwise, d_spanwise, 70.)
            return integral_velocity / (0.25 * np.pi * down_d_rotor**2), integral_turbulence



if __name__ == "__main__":
    # x_D = 5
    # C_t = 0.37
    # # C_t = 0.84
    # I_a = 0.035
    # # I_a = 0.137
    # D = 80
    # H = 70
    # y_0 = 0
    # z_0 = 1

    # # wake_plot(x_D, C_t, I_a, D, H, y=y_0, z=None)
    # # wake_plot(x_D, C_t, I_a, D, H, y=None, z=None, verbose=True)
    # turbulence_plot(x_D, C_t, I_a, D, H, y=None, z=None, verbose=True)

    # test = IshiharaWake(8, 80, 0.8, 0.077, 70)
    # print("sigma_D_r", test.sigma_D_r(500))
    # print("v_deficit", test.v_deficit(500, 0, 70))
    # print("v_wake", test.v_wake(500, 0, 70))
    # print("I_added", test.I_added(500, 0, 70))

    def turbulence(y, z):
        x_D, C, I, D = 5, 0.8, 0.077, 80
        r = np.sqrt(y**2 + z**2)

        k_star = 0.11 * (C**1.07) * (I**0.20)
        # print(k_star)
        ep = 0.23 * (C**-0.25) * (I**0.17)
        # print(ep)
        d = 2.3 * (C**-1.2)
        e = 1.0 * (I**0.1)
        f = 0.7 * (C**-3.2) * (I**-0.45)
        # print(d, e, f)
        sigma_D_r = k_star * x_D + ep
        # print(sigma_D_r)

        def k_1(r):
            return 1. if (r / D) > 0.5 else np.cos((np.pi / 2) * ((r / D) - 0.5))**2

        def k_2(r):
            return 0. if (r / D) > 0.5 else np.cos((np.pi / 2) * ((r / D) + 0.5))**2

        def delta(z, H):
            return 0. if z >= 0 else I * np.sin(np.pi * (-z / H))**2

        B = 1. / (d + e * x_D + f * ((1 + x_D)**-2))
        I_add = B * (k_1(r) * np.exp(- (((r / D) - 0.5)**2) / (2 * sigma_D_r**2)) +
                     k_2(r) * np.exp(- (((r / D) + 0.5)**2) / (2 * sigma_D_r**2))) - delta(z, 70.)
        return I_add

    R = 40.
    d = 40.

    def z_range(y, r, d_spanwise):
        assert (y >= d_spanwise - r ) and (y <= d_spanwise + r)
        return np.sqrt(r**2 - (y - d_spanwise)**2)

    # print(turbulence(50., 0.))
    # print(z_range(0, R, d))
    r, e = integrate.dblquad(turbulence, -R + d, R + d, lambda y: -z_range(y, R, d), lambda y : z_range(y, R, d))
    print("(1)-->", r / (np.pi * R**2))

    m = 5.
    n = 0.
    # T_m = None
    T_m = "Ishihara"
    I_w = None
    I_a = 0.077

    test = IshiharaWake(np.array([0., 80. * m]), 8, 0.8, 80, 70,
                          T_m=T_m, I_w=I_w, I_a=I_a)
    std = test.wake_sigma_Dr(80. * m) * 80
    # print(std)
    # a, b = test.deficit_constant(80. * m)
    # print(a, b)
    # t = test.wake_turbulence(80. * m, 40., 70.)
    # print(t)
    # v = test.wake_velocity(80. * m, 0., 70.)
    # print(v)
    iv, it = test.wake_loss(np.array([40., 0.]), 80)
    print("(2)-->", iv, it)
    # print(test.wake_turbulence_integrand(80. * m)[0](50., 70.))
