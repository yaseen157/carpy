"""Thin aerofoil theory predictions for 2D sectional drag."""
import numpy as np
from scipy.integrate import simpson

from carpy.geometry import Aerofoil
from carpy.utility import Hint, cast2numpy, point_diff
from ._common import AeroSolution

__all__ = ["ThinAerofoil2D"]
__author__ = "Yaseen Reza"


class ThinAerofoil2D(AeroSolution):
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

    def __init__(self, aerofoil: Aerofoil, alpha: Hint.num,
                 Npanels: int = None, **kwargs):
        """
        Args:
            aerofoil: Aerofoil object.
            Npanels: Number of panels to use. Optional.
        """
        # Super class call
        if Npanels is None:  # Can't add 1 to a None!
            super().__init__(aerofoil, **kwargs)
        else:
            super().__init__(aerofoil, N=Npanels + 1, **kwargs)

        Npoints = self._Nctrlpts
        panelsize = 1 / (Npoints - 1)

        # Fundamental equation of thin aerofoil theory, fourier cosine expansion
        xs = (1 - np.cos((theta := np.linspace(0, np.pi, Npoints)))) / 2
        zs = np.interp(xs, *aerofoil._camber_points(step_target=panelsize).T)
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

        A0 = float(f_A0(alpha=alpha))
        self._CL = float(self._Clalpha * (A0 + f_An(1) / 2))

        A1, A2 = f_An([1, 2])
        self._Cm = float((np.pi / 4) * (A2 - A1))

        if self.CL == 0:
            self._x_cp = np.inf
        elif (xc_cp := 0.25 * (1 + np.pi / self._CL * (A2 - A1))) < 0:
            self._x_cp = np.nan
        else:
            self._x_cp = xc_cp

        # Finish up
        self._user_readable = True
        return

    def _Cm_x(self, x: Hint.nums) -> np.ndarray:
        # Recast as necessary
        x = cast2numpy(x)

        Cm_x = self._Cm - (x - 0.25) * self._CL

        return Cm_x

    @property
    def _alpha_zl(self) -> float:
        """The angle of attack that produces zero lift."""
        panelsize = 1 / (self._Nctrlpts - 1)
        # Find points describing camber, and redistribute this in cosine spacing
        xcamb, ycamb = self.sections._camber_points(step_target=panelsize).T
        xs = (1 - np.cos((theta := np.linspace(0, np.pi, 100)))) / 2
        dz_dx = np.interp(xs, xcamb, point_diff(y=ycamb, x=xcamb))

        alpha_zl = (1 / np.pi) * simpson(dz_dx * (2 * xs), theta)
        return alpha_zl

    @property
    def _Clalpha(self) -> float:
        """Incompressible, two-dimensional thin aerofoil slope."""
        return 2 * np.pi
