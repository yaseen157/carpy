"""Methods of thin aerofoil theory."""
import numpy as np
from scipy.integrate import trapezoid

from carpy.utility import cast2numpy

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
    # camber_american = camber_british * (np.sin(theta) ** 0.5 + 0.1) / 1.1
    camber_american = camber_british * (np.sin(theta) ** 0.5)

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
