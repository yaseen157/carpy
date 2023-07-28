"""Methods of thin aerofoil theory."""
import numpy as np
from scipy.integrate import simpson, trapezoid

from carpy.utility import Hint, cast2numpy

__author__ = "Yaseen Reza"


def coords2camber(upper_points: np.ndarray,
                  lower_points: np.ndarray) -> np.ndarray:
    """
    Given upper and lower surface coordinates of an aerofoil, approximate the
    camberline.

    Args:
        upper_points: Non-dimensionalised coordinates describing the upper
            surface of an aerofoil.
        lower_points: Non-dimensionalised coordinates describing the lower
            surface of an aerofoil.

    Returns:
        Non-dimensionalised coordinates describing the camber
            line of an aerofoil.

    """
    # Recast as necessary
    upper_points = cast2numpy(upper_points)
    lower_points = cast2numpy(lower_points)

    # The number of points in self and other surfaces aren't necessarily
    # the same, so some interpolation is required
    n_points = max(len(upper_points), len(lower_points))
    theta = np.linspace(np.pi, 0, n_points)
    xnew = (np.cos(theta) + 1) / 2

    # Create functions to interpolate each surface
    # Create interpolation functions for upper and lower surfaces
    def f_upper(xs):
        """Linear function of the upper surface at each query point in xs."""
        self_interp = np.interp(xs, *upper_points.T)
        return self_interp

    def f_lower(xs):
        """Linear function of the lower surface at each query point in xs."""
        self_interp = np.interp(xs, *lower_points.T)
        return self_interp

    # Approx. American camber by weighted average of British and chordline (z=0)
    camber_british = np.mean([f_upper(xs=xnew), f_lower(xs=xnew)], axis=0)
    # camber_american = camber_british * factor that weights leading edge to y=0
    camber_american = camber_british * (np.cos(theta / 2) ** 0.25)

    return np.vstack([xnew, camber_american]).T


def coords2alphazl(camber_points: np.ndarray) -> float:
    """
    Given coordinates for the camber line of an aerofoil, predict the zero-lift
    angle of attack according to thin aerofoil theory.

    Args:
        camber_points: Non-dimensionalised coordinates describing the camber
            line of an aerofoil.

    Returns:
        The zero-lift angle of attack, in radians.

    """
    # Recast as necessary
    camber_points = cast2numpy(camber_points)

    x_c, y_c = camber_points.T
    dydx_c_midpt = np.diff(y_c) / np.diff(x_c)
    dydx_c_midpt = np.hstack([dydx_c_midpt[0], dydx_c_midpt, dydx_c_midpt[-1]])
    dydx_c = (dydx_c_midpt[1:] + dydx_c_midpt[:-1]) / 2

    # Redistribute dydx_c in the theta space
    thetas = np.linspace(0, np.pi)
    xs = (1 - np.cos(thetas)) / 2
    dydx_c = np.interp(xs, x_c, dydx_c)

    integrand = dydx_c * (2 * xs)  # == dydx_c * (1 - np.cos(thetas))
    integral = trapezoid(integrand, thetas)
    alpha_zl = integral / np.pi

    return alpha_zl


class ThinCamberedAerofoil(object):
    """
    Compute the Fourier Cosine solution for cambered aerofoils according to thin
    aerofoil theory.

    Notes:
        Theory is valid when the following assumptions are taken:

        -   Flow is incompressible
        -   Flow is irrotational
        -   Flow is inviscid (no viscosity)
        -   Angle of attack is small
        -   Aerofoil thickness is small
        -   Camber is small
        -   Drag = 0

    """

    def __init__(self, camber_points: np.ndarray, N: int = None):
        """
        Args:
            camber_points: Non-dimensionalised coordinates describing the camber
                line of an aerofoil.
            N: Number of discretisations to use in integration. Optional,
                defaults to 100.
        """
        # Recast as necessary
        camber_points = cast2numpy(camber_points)
        N = 100 if N is None else N

        # Fourier cosine solution for fundamental eq. of thin aerofoil theory
        thetas = np.linspace(0, np.pi, N)
        xs = (np.cos(thetas) + 1) / 2
        zs = np.interp(xs, *camber_points.T)
        dzdx = np.diff(zs) / np.diff(xs)
        xms = np.mean([xs[1:], xs[:-1]], axis=0)
        thetams = np.arccos(2 * xms - 1)

        def A0(alpha: np.ndarray) -> np.ndarray:
            A0 = alpha - 1 / np.pi * simpson(dzdx, xms)
            return A0

        def An(n: np.ndarray) -> np.ndarray:
            An = np.zeros(n.shape)
            for i, ni in enumerate(n.flat):
                An.flat[i] = simpson(dzdx * np.cos(n * thetams), xms)
            else:
                An *= 2 / np.pi
            return An

        self._A0 = A0
        self._An = An

        # Also compute the angle of zero lift
        self._alpha_zl = coords2alphazl(camber_points=camber_points)
        return

    def A0(self, alpha: Hint.nums) -> np.ndarray:
        """
        A component of the Fourier solution to the fundamental equation of thin
        aerofoil theory, A0=f(alpha).

        Args:
            alpha: Local angle of attack.

        Returns:
            A0.

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)
        return self._A0(alpha)

    def An(self, n: Hint.nums) -> np.ndarray:
        """
        A component of the Fourier solution to the fundamental equation of thin
        aerofoil theory, An=f(n).

        Args:
            n: The n'th element of the infinite fourier series, for n > 0.

        Returns:
            An.

        """
        # Recast as necessary
        n = cast2numpy(n)
        return self._An(n)

    def Clalpha(self, alpha: Hint.nums) -> np.ndarray:
        """
        The sectional lift-curve slope.

        Args:
            alpha: Local angle of attack.

        Returns:
            Sectional lift-curve slope.

        """
        clalpha = np.ones(cast2numpy(alpha).shape) * (2 * np.pi)
        return clalpha

    def Cl(self, alpha: Hint.nums) -> np.ndarray:
        """
        The sectional lift coefficient.

        Args:
            alpha: Local angle of attack.

        Returns:
            Sectional lift coefficient.

        """
        cl = np.pi * (2 * self.A0(alpha=alpha) + self.An(1))
        return cl

    def Cm_LE(self, alpha: Hint.nums) -> np.ndarray:
        """
        The moment coefficient at the leading edge.

        Args:
            alpha: Local angle of attack.

        Returns:
            Moment coefficient at the leading edge.

        """
        cm_le = -np.pi / 2 * (self.A0(alpha) + self.An(1) - self.An(2) / 2)
        return cm_le

    @property
    def Cm_AC(self) -> np.ndarray:
        """Moment coefficient at the aerodynamic centre (x/c=25% typically)."""
        cm_qc = np.pi / 4 * (self.An(2) - self.An(1))
        return cm_qc

    @property
    def alpha_zl(self) -> float:
        """The angle of attack that produces zero lift."""
        return self._alpha_zl
