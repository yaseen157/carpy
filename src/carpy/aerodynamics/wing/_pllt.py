"""Prandtl lifting-line theory (a.k.a Lanchester-Prandtl wing theory)."""
import numpy as np
from scipy.integrate import simpson

from carpy.utility import Hint, Quantity, cast2numpy

__all__ = ["PLLT"]
__author__ = "Yaseen Reza"


class PLLT(object):
    """
    Prandtl's lifting line (a.k.a Lanchester-Prandtl wing) theory.

    For use in incompressible, inviscid, steady flow regimes with moderate to
    high aspect ratio, unswept wings.
    """

    def __init__(
            self, span: Hint.num, alpha_inf: Hint.num,
            f_chord: Hint.func = None, f_alpha_zl: Hint.func = None,
            f_Clalpha: Hint.func = None, N: int = None):
        """
        Args:
            span: Span of the wing as viewed in planform.
            alpha_inf: Freestream angle of attack.
            f_chord: Chord as a function of spanwise position. Optional,
                defaults to a constant chord of 1 metre.
            f_alpha_zl: Zero-lift angle as a function of spanwise position.
                Optional, defaults to zero (flat plate result).
            f_Clalpha: Station lift-curve slope as a function of spanwise
                position. Optional, defaults to 2 * pi (flat plate result).
            N: Solver parameter, defines the size of an NxN matrix to solve.
                Optional, defaults to 50.
        """
        # Recast as necessary
        if f_chord is None:
            def f_chord(arg):
                """Chord length, as function of position y = [-b/2, b/2]."""
                return np.ones(cast2numpy(arg).shape)
        if f_alpha_zl is None:
            def f_alpha_zl(arg):
                """Chord length, as function of position y = [-b/2, b/2]."""
                return np.zeros(cast2numpy(arg).shape)
        if f_Clalpha is None:
            def f_Clalpha(arg):
                """Chord length, as function of position y = [-b/2, b/2]."""
                return np.ones(cast2numpy(arg).shape) * (2 * np.pi)
        N = 50 if N is None else N

        # Skip first element to prevent the generation of singular matrices
        theta0 = np.linspace(0, np.pi, N + 2)[1:-1]

        # Station parameters
        y = np.cos(theta0) * span / 2  # .. Span position
        chord = f_chord(y)  # ............. Chord length
        alpha_zl = f_alpha_zl(y)  # ....... Angle of zero-lift
        Clalpha = f_Clalpha(y)  # ......... Lift-curve slope

        # Solve matrices for Fourier series coefficients
        matA = np.zeros((N, N))
        for j in (n := np.arange(N)):
            term1 = 4 * span / Clalpha[j] / chord[j]
            term2 = (n + 1) / np.sin(theta0[j])
            matA[j] += np.sin((n + 1) * theta0[j]) * (term1 + term2)
        matB = (alpha_inf - alpha_zl)[:, None]
        matX = np.linalg.solve(matA, matB)[:, 0]  # <-- This is the slow step!!!

        self._b = Quantity(span, "m")
        self._theta0 = theta0
        self._y = Quantity(y, "m")
        self._chord = Quantity(chord, "m")
        self._matX = matX

        return

    @property
    def AR(self) -> float:
        """Wing aspect ratio, AR."""
        AR = self.b ** 2 / self.Sref
        return float(AR)

    @property
    def CL(self) -> float:
        """Wing lift coefficient, CL."""
        CL = np.pi * self.AR * self._matX[0]
        return CL

    @property
    def CLalpha(self) -> float:
        """Finite wing lift-curve slope, CLalpha."""
        a0 = 2 * np.pi
        CLalpha = a0 / (1 + a0 / (np.pi * self.AR) * (1 + self.tau))
        return CLalpha

    @property
    def Sref(self) -> Quantity:
        """Wing reference planform area, Sref."""
        # theta0 parameterisation reverses graph, so negate to get +ve area
        Sref = -simpson(y=self._chord, x=self._y)
        return Quantity(Sref, "m^{2}")

    @property
    def b(self) -> Quantity:
        """Wing span, b."""
        return self._b

    @property
    def e(self) -> float:
        """Planform span efficiency factor, e."""
        e = 1 / (1 + self.delta)
        return e

    @property
    def delta(self) -> float:
        """Elliptical lift distribution deviation factor, delta."""
        n = (np.arange(1, len(self._matX) + 1))
        delta = (n * (self._matX / self._matX[0]) ** 2)[1:].sum()
        return delta

    @property
    def tau(self) -> float:
        """Finite wing lift-curve slope deviation factor, tau."""
        n = (np.arange(1, len(self._matX) + 1))
        # n increases row-wise, theta0 is periodic column-wise
        integrand = self._chord * np.sin(n[:, None] * self._theta0)
        integral = simpson(y=integrand, x=self._y)
        tau = self.b / (2 * self.Sref) * (n * self._matX * integral)[1:].sum()
        return float(tau)
