"""
Methods of fluid drag prediction, from Hoerner.

References:
    S.F. Hoerner, "Fluid-Dynamic Drag", Hoerner Fluid Dynamics, 1965.
"""
import numpy as np

from carpy.geometry import StrandedCable
from ._common import AeroSolution

__all__ = ["HoernerCable"]
__author__ = "Yaseen Reza"


class HoernerCable(AeroSolution):
    """
    Hoerner-described method for computing the drag on a stranded cable.

    References:
        S.F. Hoerner, "Fluid-Dynamic Drag", Hoerner Fluid Dynamics, 1965,
        ch. 4, sec. 3. p. 4-5.

    """

    def __init__(self, cable: StrandedCable, **kwargs):
        # Super class call
        super().__init__(cable, **kwargs)

        # The ratio of outer strand diameter to cable diameter informs base drag
        dstrand_dc = (cable.d_strand / cable.d_cable).x
        CD0_e4 = np.interp(  # Lower Reynolds number estimate
            dstrand_dc, [0, 0.7 / 3.3, 1.3 / 3.7], [1.19, 1.14, 1.04])

        # Compute the Reynolds number, where cable diameter is the reference
        # ... find viscosity of air
        fltconditions = dict([
            ("altitude", self.altitude), ("geometric", self.geometric)])
        mu_visc = self.atmosphere.mu_visc(**fltconditions)

        # ... compute Reynolds number
        rho = self.atmosphere.rho(**fltconditions)
        Re = rho * self.TAS * cable.d_cable / mu_visc

        # The drag coefficient tends towards 1.0 at Reynolds e6
        CD0, = np.interp(Re, [1e4, 1e6], [CD0_e4, 1.0])

        # Align the force coefficient with freestream
        # ... create rotation matrix for angle of attack and of sideslip
        cos_alpha, sin_alpha = np.cos(-self.alpha), np.sin(-self.alpha)
        rot_alpha = np.array([
            [cos_alpha, 0, sin_alpha],
            [0, 1, 0],  # About the Y-axis
            [-sin_alpha, 0, cos_alpha]
        ])
        cos_beta, sin_beta = np.cos(-self.beta), np.sin(-self.beta)
        rot_beta = np.array([
            [cos_beta, -sin_beta, 0],
            [sin_beta, cos_beta, 0],
            [0, 0, 1]  # About the Z-axis
        ])
        # ... orient with the freestream
        self._CD0, self._CY, self._CL = \
            rot_beta @ rot_alpha @ np.array([CD0, 0.0, 0.0])

        # Finish up
        self._user_readable = True
        return


if __name__ == "__main__":
    mycable = StrandedCable(diameter=3e-3, Nstrands=6)
    print(HoernerCable(mycable))
