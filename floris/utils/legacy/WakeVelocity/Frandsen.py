import numpy as np

from floris.utils.model.velocity.Jensen import JensenWake

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

# The Frandsen model was originally recommended for both small and
# large regular wind farms with rectangular shapes and with spacings
# between turbines equal in both directions and lower than 10D

def _Frandsen_wake(x, v_inflow, D_r, C_t, k_w=None):
    k = 3  # either 3 Schlichting solution or 2 square root shape solution
    beta = (1 + np.sqrt(1 - C_t)) / (2 * np.sqrt(1 - C_t))
    alpha = 0.7 if k_w is None else 10 * k_w  # or 10k_w

    D_w = D_r * (((beta**(k/2)) + alpha * (x / D_r))**(1/k))
    v_deficit = 0.5 * (1 - np.sqrt(1 - 2 * ((D_r / D_w)**2) * C_t))
    v_wake = v_inflow * (1 - v_deficit)

    return D_w, v_deficit, v_wake


class FrandsenWake(JensenWake):
    def __init__(self, loc, C_t, D_r, z_hub, I_a=0.077,
                 T_m=None, I_w=None,):
        super().__init__(loc, C_t, D_r, z_hub, T_m=T_m, I_w=I_w, I_a=I_a)
        self.k = 3.  # either 3 Schlichting solution or 2 square root shape solution
        self.beta = (1 + np.sqrt(1 - C_t)) / (2 * np.sqrt(1 - C_t))
        self.alpha = 0.04 * 10. if T_m is None else (0.04 * 10. * I_w) / I_a

    def wake_width(self, x):
        return self.d_rotor * (((self.beta**(self.k / 2)) + \
            self.alpha * (x / self.d_rotor))**(1/self.k))

    def vel_deficit(self, x):
        flag = 0.5 - (1 - np.sqrt(1 - self.C_thrust))
        return 0.5 * (1 + np.sign(flag) * np.sqrt(1 - 2 * ((self.d_rotor / self.wake_width(x))**2) * self.C_thrust))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                            MISCELLANEOUS                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #





if __name__ == "__main__":
    m = 10.
    n = 0.
    I_w = 0.12

    test = FrandsenWake(np.array([0., 80. * m]), 8, 0.8, 80, 70, "Crespo", I_w=I_w)
    print(test.wake_width(80. * m))
    print(test.alpha)
    # print(test.vel_deficit(80. * m))
    # print(test.wake_velocity(500))
    d, I = test.wake_loss(np.array([80 * n, 0.]), 80)
    print(d, I)
