"""Methods for creating basic geometries."""
import numpy as np
import sympy as sp

from carpy.utility import Hint, Quantity

__all__ = ["StrandedCable"]
__author__ = "Yaseen Reza"

# ============================================================================ #
# Support Functions and Classes
# ---------------------------------------------------------------------------- #


# ----- Stranded cable diameter ------
# Place a point on the outermost point of every single strand. For Nstrands,
# the points can join into an n_sided polygon.
d_cable = sp.symbols("d_0")
d_strand = sp.symbols("d_s")
theta = sp.symbols("theta")
r_strand0 = (d_cable - d_strand) / 2

# Cosine rule for the side length, should be equal to 2 times strand radius
s0 = (2 * r_strand0 ** 2 * (1 - sp.cos(theta))) ** 0.5
s1 = d_strand
eqn_strand = sp.Eq(s0, s1)


# ============================================================================ #
# Public Functions and Classes
# ---------------------------------------------------------------------------- #

class StrandedCable(object):
    """
    Define the geometry of a stranded cable.
    """

    def __init__(self, diameter: Hint.num, Nstrands=None):
        """
        Args:
            diameter: Diameter of the cable's circumcircle.
            Nstrands: Number of strands that form the outer shell of the cable.
        """
        # Recast as necessary
        self._diameter = Quantity(diameter, "m")
        self._Nstrands = 12 if Nstrands is None else int(Nstrands)

        if self.Nstrands in [1, 2] or self.Nstrands < 0:  # Limit of geometry
            errormsg = (
                f"{self.Nstrands=} is illegal. Please choose Nstrands = 0 "
                f"(unstranded) or Nstrands > 2"
            )
            raise ValueError(errormsg)

        if self.Nstrands > 0:
            my_eqn_strand = eqn_strand.subs([
                ("d_0", self.diameter.x),
                ("theta", 2 * np.pi / self.Nstrands)]
            )
            self._d_strand = Quantity(sp.solve(my_eqn_strand), "m")
        else:
            self._d_strand = Quantity(0, "m")

        return

    @property
    def diameter(self) -> Quantity:
        """Diameter of the cable's circumcircle."""
        return self._diameter

    @property
    def length(self) -> Quantity:
        """Length of the cable."""
        return self._length

    @property
    def Nstrands(self) -> int:
        """Number of strands in the outermost layer of the cable."""
        return self._Nstrands

    @property
    def radius(self) -> Quantity:
        """Radius of the cable's circumcircle."""
        return self.diameter / 2

    @property
    def d_cable(self) -> Quantity:
        """Diameter of the cable's circumcircle."""
        return self.diameter

    @property
    def r_cable(self) -> Quantity:
        """Radius of the cable's circumcircle."""
        return self.radius

    @property
    def d_strand(self) -> Quantity:
        """Diameter of the strands that wrap around the cable."""
        return self._d_strand

    @property
    def r_strand(self) -> Quantity:
        """Radius of the strands that wrap around the cable."""
        return self.d_strand / 2
