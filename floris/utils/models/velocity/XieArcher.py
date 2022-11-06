import numpy as np
from scipy import integrate

from floris.utils.tools import power_calc_ops_old as vops


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def __Xie_Archer_wake(x, y, z, v_inflow, D_r, C_t, z_hub):
    beta = (1. + np.sqrt(1 - C_t)) / (2. * np.sqrt(1 - C_t))  # C_t < 0.9
    epsilon = 0.2 * np.sqrt(beta)
    k_y = 0.025
    k_z = 0.0175

    sigma_y_D_r = k_y * (x / D_r) + epsilon
    sigma_z_D_r = k_z * (x / D_r) + epsilon
    # r2 = (z - z_hub)**2 + (y)**2
    v_deficit = (1. - np.sqrt(1. - (C_t / (8 * sigma_y_D_r * sigma_z_D_r)))) * \
        np.exp(- (((z - z_hub)**2) / (2 * (sigma_z_D_r * D_r)**2)) -
               (((y**2) / (2 * (sigma_y_D_r * D_r)**2))))
    v_wake = v_inflow * (1 - v_deficit)

    return sigma_y_D_r, sigma_z_D_r, v_deficit, v_wake


class XieArcherWake(object):
    def __init__(self, loc, inflow, C_t, D_r, z_hub, T_m=None, I_w=None,
                 I_a=0.077):
        self.ref_loc = loc  # (x_axis, y_axis)
        self.v_inflow = inflow
        self.C_thrust = C_t
        self.d_rotor = D_r
        self.z_hub = z_hub
        self.I_a = I_a
        self.epsilon = 0.2 * np.sqrt(
            (1. + np.sqrt(1 - self.C_thrust)) / (2. * np.sqrt(1 - self.C_thrust)))

        self.T_m = None if T_m is None else vops.find_and_load_model(T_m, "tim")
        self.I_wake = None if T_m is None else I_w
        # self.k_star = 0.033 if T_m is None else 0.3837 * I_w + 0.003678
        self.k_y = 0.025 if T_m is None else (0.025 * I_w) / I_a
        self.k_z = 0.0175 if T_m is None else (0.0175 * I_w) / I_a

    def wake_sigma_Dr(self, k, x):
        return k * (x / self.d_rotor) + self.epsilon

    def deficit_constant(self, sigma_y_Dr, sigma_z_Dr):
        a = 1. - np.sqrt(1. - (self.C_thrust / (8 * sigma_y_Dr * sigma_z_Dr))) if self.C_thrust / (8 * sigma_y_Dr * sigma_z_Dr) <= 1. else 1.
        b, c = -1. / (2 * (sigma_z_Dr * self.d_rotor)**2), -1. / (2 * (sigma_y_Dr * self.d_rotor)**2)
        return a, b, c

    def wake_integrand(self, sigma_y_Dr, sigma_z_Dr, d_spanwise):
        A, B, C = self.deficit_constant(sigma_y_Dr, sigma_z_Dr)
        return lambda r, t: A * np.exp(
            (C * (r * np.cos(t) + d_spanwise)**2) + (B * (r * np.sin(t))**2)) * r

    def wake_velocity(self, x, y, z):
        sigma_y_Dr, sigma_z_Dr = self.wake_sigma_Dr(self.k_y, x), self.wake_sigma_Dr(self.k_z, x)
        v_deficit = (1. - np.sqrt(1. - (self.C_thrust / (8 * sigma_y_Dr * sigma_z_Dr)))) * \
            np.exp(- (((z - self.z_hub)**2) / (2 * (sigma_z_Dr * self.d_rotor)**2)) - (((y**2) / (2 * (sigma_y_Dr * self.d_rotor)**2))))
        return self.v_inflow * (1 - v_deficit)

    @staticmethod
    def wake_intersection(d_spanwise, y_wake, z_wake, down_d_rotor):
        return vops.wake_overlap_ellipse(d_spanwise, y_wake, z_wake, down_d_rotor)

    def wake_loss(self, down_loc, down_d_rotor, down_z_hub=None, eq=None):
        assert self.ref_loc[1] >= down_loc[1], "Reference WT must be upstream downstream WT!"
        d_streamwise,  d_spanwise = \
            np.abs(self.ref_loc[1] - down_loc[1]), np.abs(self.ref_loc[0] - down_loc[0])
        m, n, = 25., 3.  # application scope of the model and control the calculation
        sigma_y_Dr, sigma_z_Dr = \
            self.wake_sigma_Dr(self.k_y, d_streamwise), self.wake_sigma_Dr(self.k_z, d_streamwise)
        if d_spanwise > n * sigma_y_Dr * self.d_rotor or d_streamwise > m * self.d_rotor:
            return 0., 0.
        else:
            # f = lambda r, t: A * np.exp(B * ((r * np.cos(t) + d_spanwise)**2 + (r * np.sin(t))**2)) * r
            integral_velocity, _ = integrate.dblquad(
                self.wake_integrand(sigma_y_Dr, sigma_z_Dr, d_spanwise),
                0, 2 * np.pi, lambda r: 0, lambda r: down_d_rotor / 2)
            intersect_ratio = self.wake_intersection(
                    d_spanwise, 2 * sigma_y_Dr * self.d_rotor, 2 * sigma_z_Dr * self.d_rotor, down_d_rotor) \
                        if self.T_m is not None else 0. 
            I_add = self.T_m(self.C_thrust, self.I_wake, d_streamwise / self.d_rotor) \
                if self.T_m is not None else 0.
            return integral_velocity / (0.25 * np.pi * down_d_rotor**2), I_add * intersect_ratio


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #




if __name__ == "__main__":
    m = 20.
    n = 0.

    test = XieArcherWake(np.array([0., 80. * m]), 8, 0.8, 80, 70)
    std_y = test.wake_sigma_Dr(0.025, 80. * m) * 80
    std_z = test.wake_sigma_Dr(0.0175, 80. * m) * 80
    print(std_y)
    print(std_z)
    # print(test.wake_loss(np.array([std_y * n, 0.]), 80))
