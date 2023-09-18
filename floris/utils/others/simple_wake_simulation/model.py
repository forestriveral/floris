import numpy as np
from typing import Any, Dict

from tool import (
    cosd,
    sind,
)

from module import (
    TurbineGrid,
    PlanarGrid,
    FlowField,
)



class VelocityModel():
    def __init__(self) -> None:
        self.we = 0.05

    def prepare(self,
                grid: TurbineGrid | PlanarGrid,
                flow_field: FlowField,
                ) -> Dict[str, Any]:
        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
        }
        return kwargs

    def compute(self,
                 x_i: np.ndarray,
                 y_i: np.ndarray,
                 z_i: np.ndarray,
                 axial_induction_i: np.ndarray,
                 deflection_field_i: np.ndarray,
                 yaw_angle_i: np.ndarray,
                 turbulence_intensity_i: np.ndarray,
                 ct_i: np.ndarray,
                 hub_height_i,
                 rotor_diameter_i,
                 *,
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 ) -> None:

        rotor_radius = rotor_diameter_i / 2.0

        # Numexpr - do not change below without corresponding changes above.
        dx = x - x_i
        dy = y - y_i - deflection_field_i
        dz = z - z_i

        we = self.we
        NUM_EPS = 0.001

        # Construct a boolean mask to include all points downstream of the turbine
        downstream_mask = dx > 0 + NUM_EPS

        boundary_mask = np.sqrt(dy ** 2 + dz ** 2) < we * dx + rotor_radius

        # Calculate C for points within the mask and fill points outside with 0
        c = np.where(
            np.logical_and(downstream_mask, boundary_mask),
            (rotor_radius / (rotor_radius + we * dx + NUM_EPS)) ** 2,  # This is "C"
            0.0,
        )

        velocity_deficit = 2 * axial_induction_i * c

        return velocity_deficit


class DeflectionModel():
    def __init__(self) -> None:
        self.kd = 0.05
        self.ad = 0.0
        self.bd = 0.0

    def prepare(self,
                grid: TurbineGrid | PlanarGrid,
                flow_field: FlowField,
                ) -> Dict[str, Any]:
        kwargs = {
            "x": grid.x_sorted,
        }
        return kwargs

    def compute(self,
                x_i: np.ndarray,
                y_i: np.ndarray,
                yaw_i: np.ndarray,
                turbulence_intensity_i: np.ndarray,
                ct_i: np.ndarray,
                rotor_diameter_i: np.ndarray,
                *,
                x: np.ndarray,) -> None:
        # angle of deflection
        xi_init = cosd(yaw_i) * sind(yaw_i) * ct_i / 2.0
        delta_x = x - x_i

        # yaw displacement
        A = 15 * (2 * self.kd * delta_x / rotor_diameter_i + 1) ** 4.0 + xi_init ** 2.0
        B = (30 * self.kd / rotor_diameter_i)
        B = B * ( 2 * self.kd * delta_x / rotor_diameter_i + 1 ) ** 5.0
        C = xi_init * rotor_diameter_i * (15 + xi_init ** 2.0)
        D = 30 * self.kd

        yYaw_init = (xi_init * A / B) - (C / D)

        # corrected yaw displacement with lateral offset
        # This has the same shape as the grid

        deflection = yYaw_init + self.ad + self.bd * delta_x

        return deflection


class TurbulenceModel():
    def __init__(self) -> None:
        self.initial = 0.1
        self.constant = 0.5
        self.ai = 0.8
        self.downstream = -0.32

    def prepare(self) -> dict:
        pass

    def compute(self,
                ambient_TI: float,
                x: np.ndarray,
                x_i: np.ndarray,
                rotor_diameter: float,
                axial_induction: np.ndarray,
                ) -> None:
        # Replace zeros and negatives with 1 to prevent nans/infs
        delta_x = x - x_i

        # TODO: ensure that these fudge factors are needed for different rotations
        upstream_mask = delta_x <= 0.1
        downstream_mask = delta_x > -0.1

        #Keep downstream components   Set upstream to 1.0
        delta_x = delta_x * downstream_mask + np.ones_like(delta_x) * upstream_mask

        # turbulence intensity calculation based on Crespo et. al.
        ti = (self.constant
              * axial_induction ** self.ai
              * ambient_TI ** self.initial
              * (delta_x / rotor_diameter) ** self.downstream
              )
        # Mask the 1 values from above with zeros
        return ti * downstream_mask


class CombinationModel():
    def __init__(self) -> None:
        pass

    def prepare(self) -> dict:
        pass

    def compute(self,
                wake_field: np.ndarray,
                velocity_field: np.ndarray):
        """
        Combines the base flow field with the velocity defecits
        using sum of squares.

        Args:
            wake_field (np.array): The base flow field.
            velocity_field (np.array): The wake to apply to the base flow field.

        Returns:
            np.array: The resulting flow field after applying the wake to the
                base.
        """
        return np.hypot(wake_field, velocity_field)


class WakeModel():
    def __init__(self, velocity_model,
                 deflection_model,
                 turbulence_model,
                 combination_model,) -> None:
        self.velocity = velocity_model
        self.deflection = deflection_model
        self.turbulence = turbulence_model
        self.combination = combination_model