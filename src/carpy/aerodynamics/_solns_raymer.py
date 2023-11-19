"""
Methods of performance prediction from Raymer, Aircraft Design.

References:
        D.P. Raymer, "Aircraft Design: A Conceptual Approach", 6th ed., American
        Institute of Aeronautics and Astronautics, 2018.
"""
import numpy as np

from carpy.geometry import WingSections
from ._soln3d_liftingline import HorseshoeVortex

__all__ = ["RaymerSimple"]
__author__ = "Yaseen Reza"


class RaymerSimple(HorseshoeVortex):
    """
    Horseshoe Vortex Method, enhanced with a zeroth-order approximation of drag
    due to flow separation from Raymer.

    Notes:
        This method should only be used with "normal" aspect ratio wings and
        not with high-aspect-ratio designs such as with sailplanes. This method
        uses the Horseshoe Vortex Method to produce an ansatz CL.

    References:
        D.P. Raymer, "Aircraft Design: A Conceptual Approach", 6th ed., American
        Institute of Aeronautics and Astronautics, 2018, p. 444.

    """

    def __init__(self, wingsections: WingSections, **kwargs):
        # Super class call
        super().__init__(wingsections, **kwargs)

        # Make sure it's appropriate to apply this method
        # ... the wing should be mirrored
        errormsg = f"{type(self).__name__} is only for use with symmetric wings"
        assert wingsections.mirrored, errormsg

        # ... the sweep of the wing has to be simple
        errormsg = (
            f"{type(self).__name__} is designed to work on wings with simple "
            f"sweep profiles. Do not use for wings with compound/complex sweep"
        )
        sweeps = wingsections[:].sweep
        sweeps = sweeps[0:1] + sweeps[1:-1]  # Ignore last station's sweep
        assert len(set(sweeps)) == 1, errormsg

        # ... the sweep should be backward
        errormsg = f"{type(self).__name__} doesn't support forward sweep!"
        sweep = sweeps[0]
        assert sweep >= 0.0, errormsg

        # Compute Oswald span efficiency estimate
        ARfactor = (1 - 0.045 * wingsections.AR ** 0.68)
        e0_straight = 1.78 * ARfactor - 0.64
        e0_swept = 4.61 * ARfactor * np.cos(sweep) ** 0.15 - 3.1
        wt = np.interp(sweep, [0, np.radians(30)], [1, 0])
        e0_mixed = wt * e0_straight + (1 - wt) * e0_swept

        # Compute induced drag factor k, and then estimate profile drag factor
        k = self.CDi / (self.CL ** 2)
        m = 1 / (np.pi * self.sections.AR * e0_mixed) - k
        self._CD0 = m * self.CL ** 2

        return
