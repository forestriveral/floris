import numpy as np


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     LCOE                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class COE(object):
    """
    BaseCOE is the base cost of energy (COE) class that is used to determine
    the cost of energy associated with a
    :py:class:`~.optimization.scipy.layout_height.LayoutHeightOptimization`
    object.
    """

    def __init__(self, opt_obj):
        """
        Instantiate a COE model object with a LayoutHeightOptimization object.

        Args:
            opt_obj (:py:class:`~.layout_height.LayoutHeightOptimization`):
            The optimization object.
        """
        self.opt_obj = opt_obj

    # Public methods

    def FCR(self):
        """
        This method returns the fixed charge rate used in the COE calculation.

        Returns:
            float: The fixed charge rate.
        """
        return 0.079  # % - Taken from 2016 Cost of Wind Energy Review

    def TCC(self, height):
        """
        This method dertermines the turbine capital costs (TCC),
        calculating the effect of varying turbine height and rotor
        diameter on the cost of the tower. The relationship estiamted
        the mass of steel needed for the tower from the NREL Cost and
        Scaling Model (CSM), and then adds that to the tower cost
        portion of the TCC. The proportion is determined from the NREL
        2016 Cost of Wind Energy Review. A price of 3.08 $/kg is
        assumed for the needed steel. Tower height is passed directly
        while the turbine rotor diameter is pulled directly from the
        turbine object within the
        :py:class:`~.tools.floris_interface.FlorisInterface`:.

        TODO: Turbine capital cost or tower capital cost?

        Args:
            height (float): Turbine hub height in meters.

        Returns:
            float: The turbine capital cost of a wind plant in units of $/kWh.
        """
        # From CSM with a fudge factor
        tower_mass = (
            0.2694
            * height
            * (
                np.pi
                * (self.opt_obj.fi.floris.farm.turbines[0].rotor_diameter / 2) ** 2
            )
            + 1779.3
        ) / (1.341638)

        # Combo of 2016 Cost of Wind Energy Review and CSM
        TCC = 831 + tower_mass * 3.08 * self.opt_obj.nturbs / self.opt_obj.plant_kw

        return TCC

    def BOS(self):
        """
        This method returns the balance of station cost of a wind plant as
        determined by a constant factor. As the rating of a wind plant grows,
        the cost of the wind plant grows as well.

        Returns:
            float: The balance of station cost of a wind plant in units of
            $/kWh.
        """
        return 364.0  # $/kW - Taken from 2016 Cost of Wind Energy Review

    def FC(self):
        """
        This method returns the finance charge cost of a wind plant as
        determined by a constant factor. As the rating of a wind plant grows,
        the cost of the wind plant grows as well.

        Returns:
            float: The finance charge cost of a wind plant in units of $/kWh.
        """
        return 155.0  # $/kW - Taken from 2016 Cost of Wind Energy Review

    def O_M(self):
        """
        This method returns the operational cost of a wind plant as determined
        by a constant factor. As the rating of a wind plant grows, the cost of
        the wind plant grows as well.

        Returns:
            float: The operational cost of a wind plant in units of $/kWh.
        """
        return 52.0  # $/kW - Taken from 2016 Cost of Wind Energy Review

    def COE(self, height, AEP_sum):
        """
        This method calculates and returns the cost of energy of a wind plant.
        This cost of energy (COE) formulation for a wind plant varies based on
        turbine height, rotor diameter, and total annualized energy production
        (AEP). The components of the COE equation are defined throughout the
        BaseCOE class.

        Args:
            height (float): The hub height of the turbines in meters
                (all turbines are set to the same height).
            AEP_sum (float): The annualized energy production (AEP)
                for the wind plant as calculated across the wind rose
                in kWh.

        Returns:
            float: The cost of energy for a wind plant in units of
            $/kWh.
        """
        # Comptue Cost of Energy (COE) as $/kWh for a plant
        return (
            self.FCR() * (self.TCC(height) + self.BOS() + self.FC()) + self.O_M()
        ) / (AEP_sum / 1000 / self.opt_obj.plant_kw)



def LCOE(layout, powers, capacity, **kwargs):
    # LCOE: (CAPEX * CRF + OPEX_annual) / AEP
    
    # CAPEX = CAPEX_wt * C * N_wt + sum(C_fd_i)
    # CAPEX: the capital expenditure [€]
    # CAPEX_wt: the capital cost per MW of the WTs [M€/MW]
    # C: WT’s capacity  [MW]
    # N_wt: the number of WTs
    # C_fd_i: the cost of the i-th WT’s foundation  [M€/MW]
    
    # CRF: the capital recovery factor
    # OPEX_annual: the annualized operation and maintenance (O&M) cost [€/year]
    # AEP: the annual energy production of the WF  [MWh]
    watdepth = eval(kwargs.get("wdepth", "constant"))
    
    CAPEX_wt = 3.5  # [M€/MW]
    # depths = depth_with_offshore_distance(xs)
    # found_cost = np.sum(np.vectorize(foundation_cost)(watdepth))
    assert layout.shape[0] == capacity.shape[0]
    found_cost = np.sum(np.vectorize(foundation_cost)(watdepth(layout)) * capacity)
    CAPEX = CAPEX_wt * np.sum(capacity) + found_cost
    
    # a function of the discount rate r [%] and the WF’s lifetime N [year].
    r = 0.052   # discount rate
    N = 25  # wind farm lifetime
    CRF = r / (1 - (1 + r)**(- N))
    
    # OPEX_annual = OPEX_unit * C * N_wt (deprecated)
    # OPEX_annual = OPEX_unit * np.sum(capacity) * 1.0e-3  (deprecated)
    # OPEX_annual = OPEX_ref * Capacity * [ 1 + 0.5 * (CF - CF_ref)]
    # OPEX_unit: the annual O&M cost of unit electricity  [€/kW/year]
    OPEX_unit, CF_ref = 106 * 1.0e-3, 0.4  # [€/kW/year]
    CF = np.sum(powers[:, 0]) / np.sum(capacity)
    OPEX_annual = OPEX_unit * np.sum(capacity) * ( 1 + 0.5 * (CF - CF_ref))
    
    T = 8760 #  working hours per year [h]
    AEP = T * np.sum(powers[:, 0]) * 1.0e-6
    
    # print("CAPEX * CRF: ", CAPEX * CRF)
    # print("OPEX_annual: ", OPEX_annual)
    # print("ratio: ", CAPEX * CRF / OPEX_annual)
    # print("AEP: ", AEP)
    
    LCOE = (CAPEX * CRF + OPEX_annual) / AEP  # [M€/MWh/year]
    
    return LCOE   # [€/MWh/year]


def foundation_cost(depth):
    # water depth: (0 m, 100 m)
    assert depth <= 100. and depth >= 0.
    C_f_shallow = 0.15 + 1.0e-5 * depth**3
    C_f_intermediate = 0.35 + 0.4e-5 * depth**3
    C_f_deep = 0.35 + 0.016 * depth
    if depth <= 20:
        return C_f_shallow
    elif depth < 40. and depth > 20.:
        return min(C_f_intermediate, C_f_shallow)
    elif depth <= 50. and depth >= 40.:
        return C_f_intermediate
    elif depth < 70. and depth > 50.:
        return min(C_f_intermediate, C_f_deep)
    else:
        return C_f_deep


def constant(layout):
    return 10.


def linear_x(layout):
    # 0.001 * x + 12
    return - 0.003 * layout[:, 0] + 21


def linear_y(layout):
    # 0.001 * x + 12
    return - 0.003 * layout[:, 1] + 21

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                   OBJECTIVE                                  #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def maxpower(layout, powers, capacity, **kwargs):
    return np.sum(powers[:, 0])


def mosetti(layout, powers, capacity, **kwargs):
    P_tot, N  = np.sum(powers[:, 0]), layout.shape[0]
    cost = N * ((2 / 3) + (1 / 3) * np.exp(-0.00174 * N**2))
    return cost / P_tot


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #








if __name__ == "__main__":
    
#    print(np.mean(linear_x(np.array([[0, 0], [5040, 5040]]))))
    N = np.array([25, 36, 49])
    cost = N * ((2 / 3) + (1 / 3) * np.exp(-0.00174 * N**2))
    print(cost)
    power = np.array([28.43, 39.78, 53.57])
    coe = np.array([0.711, 0.647, 0.630])
    print(cost / power)
    print(cost / coe)