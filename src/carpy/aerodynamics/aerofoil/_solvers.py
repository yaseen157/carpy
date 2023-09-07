"""A module of various methods used to estimate aerofoil parameters."""
from carpy.utility import Hint, Quantity

__all__ = ["ThinAerofoils", "InviscidPanel"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Support Functions and Classes
# ---------------------------------------------------------------------------- #

class AerofoilSolution(object):
    """
    Template object, of which solvers should aim to produce all attributes of.
    """
    _Cd: float
    _Cl: float
    _Clalpha: float
    _Cm: Hint.func
    _Cm_ac: float
    _x_ac: float
    _x_cp: float
    _alpha_zl: float

    def __str__(self):
        params2print = \
            ["Cd", "Cl", "Clalpha", "sref", "x_ac", "x_cp", "alpha_zl"]

        return_string = f"{type(self).__name__}\n"
        for param in params2print:
            return_string += f">\t{param:<8} = {getattr(self, param)}\n"

        return return_string

    @property
    def Cd(self) -> float:
        """Sectional pressure drag coefficient."""
        return self._Cd

    @property
    def Cl(self) -> float:
        """Sectional lift coefficient."""
        return self._Cl

    @property
    def Clalpha(self) -> float:
        """Sectional lift-curve slope."""
        return self._Clalpha

    @property
    def Cm(self) -> Hint.func:
        """Sectional moment coefficient, at a given position along the chord."""
        return self._Cm

    @property
    def Cm_ac(self) -> float:
        """Sectional moment coefficient, at the aerofoil aerodynamic centre."""
        return self._Cm_ac

    @property
    def sref(self) -> Quantity:
        """Sectional area, given a reference chord length of 1."""
        return self._sref

    @property
    def x_ac(self) -> float:
        """Chordwise position of the aerofoil's aerodynamic centre."""
        return self._x_ac

    @property
    def x_cp(self) -> float:
        """Chordwise position of the aerofoil's centre of pressure."""
        return self._x_cp

    @property
    def alpha_zl(self) -> float:
        """The aerofoil's angle (of attack) of zero-lift."""
        return self._alpha_zl


# ============================================================================ #
# Public classes
# ---------------------------------------------------------------------------- #

class ThinAerofoils(AerofoilSolution):
    """
    Thin and cambered aerofoil theory.

    For use when the following assumptions are taken:

    -   Flow is incompressible
    -   Flow is irrotational
    -   Flow is inviscid (no viscosity)
    -   Angle of attack is small
    -   Aerofoil thickness is small
    -   Camber is small
    -   Drag = 0

    """

    def __init__(self, aerofoil, alpha: Hint.num):
        return


class InviscidPanel(AerofoilSolution):
    pass
