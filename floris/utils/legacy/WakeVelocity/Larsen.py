import numpy as np


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def Larsen_wake(x, r, v_inflow, D_r, C_T, I_a, z_hub):
    D_nb = max(1.08 * D_r, 1.08 * D_r + 21.7 * D_r * (I_a - 0.05))
    print(D_nb)
    D_95 = D_nb + min(z_hub, D_nb)
    print(D_95)
    x_0 = ((9.5 * D_r) / ((D_95 / D_r)**3)) - 1
    print(x_0)
    c_1 = ((D_r / 2)**-0.5) * ((C_T * 0.25 * np.pi * D_r**2 * x_0)**(-5/6))
    print(c_1)
    # 138.48000000000002
    # 208.48000000000002
    # 41.942799982038316
    # 6.968226939885172e-06

    D_w = 2 * (((35 * 3 * c_1**2) / (2 * np.pi))**(1/5)) * \
        ((C_T * 0.25 * np.pi * D_r**2 * x)**(1/3))
    print(D_w)
    v_deficit = (-1 / 9) * ((C_T * 0.25 * np.pi * D_r**2 * (x + x_0)**-2)**(1/3)) * \
        ((r**(3/2) * ((3 * (c_1**2) * C_T * 0.25 * np.pi * D_r**2 * (x + x_0)**-2)**(-1/2))) -
            (((35 / (2 * np.pi))**(3/10)) * ((3 * c_1**2)**(-1/5))))**2
    print(v_deficit)
    v_wake = v_inflow * (1 - v_deficit)
    return D_w, v_deficit, v_wake


class LarsenWake(object):
    def __init__(self, v_inflow, D_r, C_T, I_a, z_hub):
        self.v_inflow = v_inflow
        self.d_rotor = D_r
        self.c_thrust = C_T
        self.I_ambient = I_a  #  assumed to be always greater than 5%
        self.z_hub = z_hub

        self._D_nb = max(1.08 * self.d_rotor, 1.08 *
                         self.d_rotor + 21.7 * self.d_rotor * (self.I_ambient - 0.05))
        self._D_95 = self._D_nb + min(self.z_hub, self._D_nb)
        self._x_0 = ((9.5 * self.d_rotor) / ((self._D_95 / self.d_rotor)**3)) - 1
        self._c_1 = ((self.d_rotor / 2)**-0.5) * \
            ((C_T * 0.25 * np.pi * self.d_rotor**2 * self._x_0)**(-5/6))

        self.d_wake = self._wake_width
        self.v_deficit = self._velocity_deficit
        self.v_wake = self._velocity_wake

    def _wake_width(self, x):
        return 2 * (((35 * 3 * self._c_1**2) / (2 * np.pi))**(1/5)) * ((self.c_thrust * 0.25 * np.pi * self.d_rotor**2 * x)**(1/3))

    def _velocity_deficit(self, x, r):
        return (-1 / 9) * ((self.c_thrust * 0.25 * np.pi * self.d_rotor**2 * (x + self._x_0)**-2)**(1/3)) * \
        ((r**(3/2) * ((3 * (self._c_1**2) * self.c_thrust * 0.25 * np.pi * self.d_rotor**2 * (x + self._x_0)**-2)**(-1/2))) -
            (((35 / (2 * np.pi))**(3/10)) * ((3 * self._c_1**2)**(-1/5))))**2

    def _velocity_wake(self, x, r):
        return self.v_inflow * (1 - self.v_deficit(x, r))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                            MISCELLANEOUS                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #





if __name__ == "__main__":
    # test = LarsenWake(8, 80, 0.8, 0.08, 70)
    # print(test.d_wake(500))
    # print(test.v_deficit(500, 150))
    # print(test.v_wake(500, 150))

    Larsen_wake(500, 150, 8, 80, 0.8, 0.08, 70)
