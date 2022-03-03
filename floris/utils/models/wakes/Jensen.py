import math
import time
import numpy as np
from scipy import integrate

from floris.utils.tools import valid_ops as vops


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class JensenWake(object):

    def __init__(self, loc, C_t, D_r, z_hub, I_a=0.077, ytheta=0.,
                 T_m=None, I_w=None,):
        if C_t < 0:
            print("Thrust", C_t)
        assert C_t > 0, "Invalid thrust factor."
        if T_m is not None:
            assert not math.isnan(I_w), "Invalid wake turbulence."
        assert I_a > 0, "Invalid ambient turbulence."

        self.ref_loc = loc  # (x_axis, y_axis)
        self.C_thrust = C_t
        self.d_rotor = D_r
        self.z_hub = z_hub
        self.I_a = I_a

        self.T_m = T_m
        self.I_wake = None if T_m is None else I_w
        self.k_wake = 0.05 if T_m is None else (0.05 * I_w) / I_a
        self.x_D_ex = self.wake_exclusion
        self.ytheta = ytheta

    def wake_width(self, x):
        return self.d_rotor + 2 * self.k_wake * x

    def vel_deficit(self, x):
        return (1 - np.sqrt(1 - self.C_thrust)) * (self.d_rotor / self.wake_width(x))**2

    def wake_velocity(self, inflow, x):
        return inflow * (1 - self.vel_deficit(x))

    def wake_region(self, dist):
        if self.ref_loc is None:
            raise ValueError(
                "Reference location is invalid. Need to initiate first.")
        else:
            if self.ref_loc.shape != (2, ):
                raise ValueError(
                    "Reference location format error!")
            else:
                r_wake = self.wake_width(dist) / 2
                assert r_wake >= 0 and r_wake != np.NaN, "Invalid wake diameter."
                return r_wake

    @staticmethod
    def wake_intersection(d_spanwise, r_wake, down_d_rotor):
        return vops.wake_overlap(d_spanwise, r_wake, down_d_rotor)

    @property
    def wake_exclusion(self, m=0.01):
        return (np.sqrt((1 - np.sqrt(1 - self.C_thrust)) / m) - 1) / (2 * self.k_wake)

    def wake_loss(self, down_loc, down_d_rotor):
        # determine whether the downstream wind turbine inside the wake region
        assert self.ref_loc[1] >= down_loc[1], "Reference WT must be upstream downstream WT!"
        d_streamwise,  d_spanwise = \
            np.abs(self.ref_loc[1] - down_loc[1]), np.abs(self.ref_loc[0] - down_loc[0])
        if d_streamwise == 0.:
            return 0., 0.
        if (d_streamwise / self.d_rotor) < self.x_D_ex:
            wake_offset = self.wake_offset(self.ytheta, d_streamwise)
            if self.ref_loc[0] - down_loc[0] == 0:
                d_spanwise = np.abs(wake_offset)
            elif self.ref_loc[0] - down_loc[0] > 0:
                d_spanwise = np.abs(d_spanwise + wake_offset) if wake_offset >= 0 \
                    else np.abs(np.abs(d_spanwise) - np.abs(wake_offset))
            else:
                d_spanwise = np.abs(np.abs(d_spanwise) - np.abs(wake_offset)) \
                    if wake_offset >= 0 else np.abs(d_spanwise + wake_offset)
            r_wake, vel_deficit = \
                self.wake_region(d_streamwise), self.vel_deficit(d_streamwise)
            intersect_ratio = self.wake_intersection(d_spanwise, r_wake, down_d_rotor)
            # if intersect_ratio != 0.:
            #     data = [d_spanwise, r_wake, down_d_rotor, intersect_ratio]
            #     JensenWake.data_recorder.append[data]
            # print("intersect_ratio:", intersect_ratio)
            # I_add = Crespo(
            #     (1 - np.sqrt(1 - self.C_thrust)) / 2, self.I_wake, d_streamwise / self.d_rotor)
            I_add = self.T_m(self.C_thrust, self.I_wake, d_streamwise / self.d_rotor) \
                if self.T_m is not None else 0.
            return vel_deficit * intersect_ratio, I_add * intersect_ratio
        else:
            return 0., 0.

    def wake_offset(self, ytheta, distance):
        ytheta, distance = ytheta / 360 * 2 * np.pi, distance / self.d_rotor
        theta_func = lambda x_D: np.tan(
            np.cos(ytheta)**2 * np.sin(ytheta) * self.C_thrust * 0.5 * (1 + 0.09 * x_D)**-2)
        offset = integrate.quad(theta_func, 0, distance)[0] * self.d_rotor
        return offset


def Jensen_data_generator(num=500):
    # 40 > r_wake - d_spanwise > - 40
    # r_wake: (40, 500)
    # d_spanwise: (r_wake - 40, r_wake + 40)
    data = np.zeros((num, 4))
    rotor = np.random.uniform(30, 50, num)
    r_wake = np.random.uniform(rotor, 2 * rotor, num)
    d_spanwise = np.random.uniform(r_wake - rotor, r_wake + rotor, num)
    data[:, 0], data[:, 1], data[:, 2] = rotor, r_wake, d_spanwise
    data[:, -1] = np.vectorize(vops.wake_overlap)(data[:, 1], data[:, 2], data[:, 0])
    return data


def calculation_test(num=500, test="vect"):
    data = np.zeros((num, 4))
    rotor = np.random.uniform(30, 50, num)
    r_wake = np.random.uniform(rotor, 2 * rotor, num)
    d_spanwise = np.random.uniform(r_wake - rotor, r_wake + rotor, num)
    data[:, 0], data[:, 1], data[:, 2] = rotor, r_wake, d_spanwise
    start = time.time()
    if test == "vect":
        data[:, -1] = np.vectorize(vops.wake_overlap)(data[:, 1], data[:, 2], data[:, 0])
    if test == "iter":
        for i in range(num):
            data[i, -1] = vops.wake_overlap(data[i, 1], data[i, 2], data[i, 0])
    end = time.time()
    print(f"{test} | Using time: {end - start}")
    
    return data



if __name__ == "__main__":
    m = 10.
    n = 0.
    I_w = 0.12
    
    # test = JensenWake(np.array([0., 80. * m]), 8, 0.8, 80, 70,
    #                   I_w=I_w)
    # print(test.wake_width(80. * m))
    # print(test.vel_deficit(80. * m))
    # print(test.wake_velocity(500))
    # d, I = test.wake_loss(np.array([80 * n, 0.]), 80)
    # print(d, I)
    
    # data = Jensen_data_generator(50)
    # print(data.shape)
    
    calculation_test(num=50000, test="vect")
    calculation_test(num=50000, test="iter")