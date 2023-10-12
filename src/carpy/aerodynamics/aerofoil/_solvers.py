"""A module of various methods used to estimate aerofoil parameters."""
import numpy as np
from scipy.integrate import simpson

from carpy.utility import Hint, cast2numpy, point_diff

__all__ = ["ThinAerofoil"]
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
        params = ["Cd", "Cl", "Clalpha", "Cm_ac", "x_ac", "x_cp", "alpha_zl"]

        return_string = f"{type(self).__name__}\n"
        for param in params:
            if hasattr(self, f"_{param}"):
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

class ThinAerofoil(AerofoilSolution):
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

    References:
        -   Fundamentals of Aerodynamics 6th Ed. Chapter 4, John D. Anderson Jr.
    """

    def __init__(self, aerofoil, alpha: Hint.num, N: int = None):
        """
        Args:
            aerofoil: Aerofoil object.
            alpha: Angle of attack.
            N: Number of discretised points in the aerofoil's camber line.
        """
        # Recast as necessary
        N = 100 if N is None else int(N)

        # Fundamental equation of thin aerofoil theory, fourier cosine expansion
        xs = (1 - np.cos((theta := np.linspace(0, np.pi, N)))) / 2
        zs = np.interp(xs, *aerofoil._camber_points(step_target=0.05).T)
        dz_dx = point_diff(y=zs, x=xs)

        def f_A0(alpha: np.ndarray) -> np.ndarray:
            """
            Element A0 of the Fourier cosine solution to the fundamental
            equation of thin aerofoil theory.

            Args:
                alpha: Angle of attack.

            Returns:
                Coefficient A0 of the Fourier cosine solution.

            """
            A0 = alpha - 1 / np.pi * simpson(dz_dx, xs)
            return A0

        def f_An(n: Hint.nums) -> np.ndarray:
            """
            Element(s) An of the Fourier cosine solution to the fundamental
            equation of thin aerofoil theory.

            Args:
                n: The n'th element of the infinite fourier series, for n > 0.

            Returns:
                Coefficient(s) An of the Fourier cosine solution.

            """
            # Recast as necessary
            n = cast2numpy(n)

            An = np.zeros(n.shape)
            for i, ni in enumerate(n.flat):
                An.flat[i] = simpson(dz_dx * np.cos(ni * theta), theta)
            else:
                An *= 2 / np.pi
            return An

        self._aerofoil = aerofoil
        self._A0 = float(f_A0(alpha=alpha))
        self._f_An = f_An

        return

    @property
    def _Cd(self) -> float:
        """The sectional drag coefficient."""
        return 0

    @property
    def _Cl(self) -> float:
        """The sectional lift coefficient."""
        cl = float(self._Clalpha * (self._A0 + self._f_An(1) / 2))
        return cl

    @property
    def _Clalpha(self) -> float:
        """The sectional lift-curve slope."""
        a0 = 2 * np.pi
        return a0

    def _Cm(self, x: Hint.nums) -> np.ndarray:
        # Recast as necessary
        x = cast2numpy(x)

        Cm = self._Cm_ac - (x - 0.25) * self._Cl

        return Cm

    @property
    def _Cm_ac(self) -> float:
        """Moment coefficient at the aerodynamic centre."""
        A1, A2 = self._f_An([1, 2])
        cm_ac = float((np.pi / 4) * (A2 - A1))
        return cm_ac

    @property
    def _x_ac(self) -> float:
        """The chordwise position of the aerodynamic centre."""
        return 0.25

    @property
    def _x_cp(self) -> float:
        """The chordwise position of the centre of pressure."""
        A1, A2 = self._f_An([1, 2])
        xc_cp = 0.25 * (1 + np.pi / self._Cl * (A2 - A1))

        if xc_cp < 0:
            return np.nan

        return xc_cp

    @property
    def _alpha_zl(self) -> float:
        """The angle of attack that produces zero lift."""
        # Find points describing camber, and redistribute this in cosine spacing
        xcamber, ycamber = self._aerofoil._camber_points(step_target=0.05).T
        xs = (1 - np.cos((theta := np.linspace(0, np.pi, 100)))) / 2
        dz_dx = np.interp(xs, xcamber, point_diff(y=ycamber, x=xcamber))

        alpha_zl = (1 / np.pi) * simpson(dz_dx * (2 * xs), theta)
        return alpha_zl
