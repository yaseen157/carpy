"""
Vortex-source panel method for the pressure distribution around an aerofoil.

The code contained in this file is originally sourced from Aero Python (see
References section), and has been modified for improved speed and compatibility.

References:
    -   Barba, Lorena A., and Mesnard Olivier (2019). Aero Python: classical
        aerodynamics of potential flow using Python. Journal of Open Source
        Education, 2(15), 45, https://doi.org/10.21105/jose.00045.
    -   NACA0012 validation case:
        https://turbmodels.larc.nasa.gov/naca0012_val.html

"""
from functools import cached_property

import numpy as np
import scipy.integrate as sint

from carpy.geometry import Aerofoil
from carpy.utility import Hint, cast2numpy
from ._common import AeroSolution

__all__ = ["VortexSource2D"]
__author__ = "Yaseen Reza"


class Panel(object):
    """
    Contains information related to a panel.
    """

    def __init__(self, xa: Hint.num, ya: Hint.num, xb: Hint.num, yb: Hint.num):
        """
        Initialise a panel (aerodynamic wall).

        Sets the panels end-points and calculates various features of the panel.
        Initialises source-strength, tangential velocity and pressure
        coefficient to zero.

        Args:
            xa: x-coordinate of the first end-point.
            ya: y-coordinate of the first end-point.
            xb: x-coordinate of the second end-point.
            yb: y-coordinate of the second end-point.
        """

        self.xa, self.ya = xa, ya  # panel starting-point
        self.xb, self.yb = xb, yb  # panel ending-point

        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2  # panel center
        self.length = np.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)  # panel length

        # orientation of panel (angle between x-axis and panel's normal)
        if xb - xa <= 0.0:
            self.beta = np.arccos((yb - ya) / self.length)
        else:
            self.beta = np.pi + np.arccos(-(yb - ya) / self.length)

        # panel location
        if self.beta <= np.pi:
            self.loc = 'upper'  # upper surface
        else:
            self.loc = 'lower'  # lower surface

        self.sigma = 0.0  # source strength
        self.vt = 0.0  # tangential velocity
        self.cp = 0.0  # pressure coefficient
        return

    @cached_property
    def cos_beta(self):
        """Pre-computed and cached result of the cosine of panel angle beta."""
        return np.cos(self.beta)

    @cached_property
    def sin_beta(self):
        """Pre-computed and cached result of the sine of panel angle beta."""
        return np.sin(self.beta)


def define_panels(aerofoil, N: int = None) -> np.ndarray:
    """
    Discretizes the geometry into panels using 'cosine' method.

    Args:
        aerofoil: The aerofoil geometry object to be broken into panels.
        N: Number of panels. Optional, defaults to 40.

    Returns:
        panels: 1D Numpy array of Panel objects.
            The list of panels.

    """
    # Recast as necessary
    points = aerofoil.section.geometry.points
    x, y = cast2numpy(points).T
    N = 40 if N is None else N

    R = (x.max() - x.min()) / 2.0  # circle radius
    x_center = (x.max() + x.min()) / 2.0  # x-coordinate of circle center

    theta = np.linspace(0.0, 2.0 * np.pi, N + 1)  # array of angles
    x_circle = x_center + R * np.cos(theta)  # x-coordinates of circle

    x_ends = np.copy(x_circle)  # x-coordinate of panels end-points
    y_ends = np.empty_like(x_ends)  # y-coordinate of panels end-points

    # extend coordinates to consider closed surface
    x, y = np.append(x, x[0]), np.append(y, y[0])

    def approx_bounded(a, x, b):
        """Function that evaluates whether x lies in the bounds [a, b]."""
        dir1 = (a <= x or np.isclose(a, x)) and (x <= b or np.isclose(x, b))
        dir2 = (b <= x or np.isclose(b, x)) and (x <= a or np.isclose(x, a))
        return dir1 | dir2

    # compute y-coordinate of end-points by projection
    infl = 0
    for i in range(N + 1):  # Iterate over panel end points that need computing
        while infl < len(x) - 1:  # Iterate over source coordinates
            if approx_bounded(x[infl], x_ends[i], x[infl + 1]):
                # If x_diff is zero, that means two coordinates with the same
                # abscissa. To not duplicate the 'y' value, 'I' must advance 1.
                if i > 0 and np.diff(x_ends)[i - 1] == 0:
                    infl += 1  # <-- implies next bounds set includes x_ends[i]
                break
            else:
                infl += 1
        # Linear interpolation
        a = (y[infl + 1] - y[infl]) / (x[infl + 1] - x[infl])
        b = y[infl + 1] - a * x[infl + 1]
        y_ends[i] = a * x_ends[i] + b

    # create panels
    panels = np.empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])

    return panels


class Freestream:
    """
    Freestream conditions.
    """

    def __init__(self, u_inf=1.0, alpha=0.0):
        """
        Sets the freestream speed and angle.

        Parameters
        ----------
        u_inf: float, optional
            Freestream speed;
            default: 1.0.
        alpha: float, optional
            Angle of attack;
            default 0.0.
        """
        self.u_inf = u_inf
        self.alpha = float(alpha)

    @cached_property
    def cos_alpha(self):
        """Pre-computed and cached result of the cosine of angle of attack."""
        return np.cos(self.alpha)

    @cached_property
    def sin_alpha(self):
        """Pre-computed and cached result of the sine of angle of attack."""
        return np.sin(self.alpha)


def integral(x, y, panel, dxdk, dydk):
    """
    Evaluates the contribution from a panel at a given point.

    Parameters
    ----------
    x: float
        x-coordinate of the target point.
    y: float
        y-coordinate of the target point.
    panel: Panel object
        Panel whose contribution is evaluated.
    dxdk: float
        Value of the derivative of x in a certain direction.
    dydk: float
        Value of the derivative of y in a certain direction.

    Returns
    -------
    Contribution from the panel at a given point (x, y).
    """

    def integrand(s):
        """Given arc length s, find influence of panel on target point."""
        return (((x - (panel.xa - panel.sin_beta * s)) * dxdk +
                 (y - (panel.ya + panel.cos_beta * s)) * dydk) /
                ((x - (panel.xa - panel.sin_beta * s)) ** 2 +
                 (y - (panel.ya + panel.cos_beta * s)) ** 2))

    return sint.quad(integrand, 0.0, panel.length)[0]


def source_contribution_normal(panels):
    """
    Builds the source contribution matrix for the normal velocity.

    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.

    Returns
    -------
    A: 2D Numpy array of floats
        Source contribution matrix.
    """
    A = np.empty((panels.size, panels.size), dtype=float)
    # source contribution on a panel from itself
    np.fill_diagonal(A, 0.5)
    # source contribution on a panel from others
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / np.pi * integral(panel_i.xc, panel_i.yc,
                                                 panel_j,
                                                 panel_i.cos_beta,
                                                 panel_i.sin_beta)
    return A


def vortex_contribution_normal(panels):
    """
    Builds the vortex contribution matrix for the normal velocity.

    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.

    Returns
    -------
    A: 2D Numpy array of floats
        Vortex contribution matrix.
    """
    A = np.empty((panels.size, panels.size), dtype=float)
    # vortex contribution on a panel from itself
    np.fill_diagonal(A, 0.0)
    # vortex contribution on a panel from others
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = -0.5 / np.pi * integral(panel_i.xc, panel_i.yc,
                                                  panel_j,
                                                  panel_i.sin_beta,
                                                  -panel_i.cos_beta)
    return A


def kutta_condition(A_source, B_vortex):
    """
    Builds the Kutta condition array.

    Parameters
    ----------
    A_source: 2D Numpy array of floats
        Source contribution matrix for the normal velocity.
    B_vortex: 2D Numpy array of floats
        Vortex contribution matrix for the normal velocity.

    Returns
    -------
    b: 1D Numpy array of floats
        The left-hand side of the Kutta-condition equation.
    """
    b = np.empty(A_source.shape[0] + 1, dtype=float)
    # matrix of source contribution on tangential velocity
    # is the same than
    # matrix of vortex contribution on normal velocity
    b[:-1] = B_vortex[0, :] + B_vortex[-1, :]
    # matrix of vortex contribution on tangential velocity
    # is the opposite of
    # matrix of source contribution on normal velocity
    b[-1] = - np.sum(A_source[0, :] + A_source[-1, :])
    return b


def build_singularity_matrix(A_source, B_vortex):
    """
    Builds the left-hand side matrix of the system
    arising from source and vortex contributions.

    Parameters
    ----------
    A_source: 2D Numpy array of floats
        Source contribution matrix for the normal velocity.
    B_vortex: 2D Numpy array of floats
        Vortex contribution matrix for the normal velocity.

    Returns
    -------
    A:  2D Numpy array of floats
        Matrix of the linear system.
    """
    A = np.empty((A_source.shape[0] + 1, A_source.shape[1] + 1), dtype=float)
    # source contribution matrix
    A[:-1, :-1] = A_source
    # vortex contribution array
    A[:-1, -1] = np.sum(B_vortex, axis=1)
    # Kutta condition array
    A[-1, :] = kutta_condition(A_source, B_vortex)
    return A


def build_freestream_rhs(panels, freestream):
    """
    Builds the right-hand side of the system
    arising from the freestream contribution.

    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    freestream: Freestream object
        Freestream conditions.

    Returns
    -------
    b: 1D Numpy array of floats
        Freestream contribution on each panel and on the Kutta condition.
    """
    b = np.empty(panels.size + 1, dtype=float)
    # freestream contribution on each panel
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * np.cos(freestream.alpha - panel.beta)
    # freestream contribution on the Kutta condition
    b[-1] = -freestream.u_inf * (np.sin(freestream.alpha - panels[0].beta) +
                                 np.sin(freestream.alpha - panels[-1].beta))
    return b


def compute_tangential_velocity(panels, freestream, gamma, A_source, B_vortex):
    """
    Computes the tangential surface velocity.

    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    freestream: Freestream object
        Freestream conditions.
    gamma: float
        Circulation density.
    A_source: 2D Numpy array of floats
        Source contribution matrix for the normal velocity.
    B_vortex: 2D Numpy array of floats
        Vortex contribution matrix for the normal velocity.
    """
    A = np.empty((panels.size, panels.size + 1), dtype=float)
    # matrix of source contribution on tangential velocity
    # is the same than
    # matrix of vortex contribution on normal velocity
    A[:, :-1] = B_vortex
    # matrix of vortex contribution on tangential velocity
    # is the opposite of
    # matrix of source contribution on normal velocity
    A[:, -1] = -np.sum(A_source, axis=1)
    # freestream contribution
    b = freestream.u_inf * np.sin([freestream.alpha - panel.beta
                                   for panel in panels])

    strengths = np.append([panel.sigma for panel in panels], gamma)

    tangential_velocities = np.dot(A, strengths) + b

    for i, panel in enumerate(panels):
        panel.vt = tangential_velocities[i]


def compute_pressure_coefficient(panels, freestream):
    """
    Computes the surface pressure coefficients.

    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    freestream: Freestream object
        Freestream conditions.
    """
    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf) ** 2


class VortexSource2D(AeroSolution):
    """
    Given an aerofoil object, angle of attack, and number of panels to
    discretise the geometry, compute pressure distribution data.
    """

    def __init__(self, aerofoil: Aerofoil, Npanels: Hint.num = None, **kwargs):
        """
        Args:
            aerofoil: Aerofoil object.
            Npanels: Number of discretised points in the aerofoil's surface.
                Optional, defaults to 100.

        """
        # discretize geoemetry into panels
        self.panels = define_panels(aerofoil, N=Npanels)

        # Super class call
        super().__init__(aerofoil, N=len(self.panels) + 1, **kwargs)

        # Compute source and vortex influence matrices
        A_source = source_contribution_normal(self.panels)
        B_vortex = vortex_contribution_normal(self.panels)

        # define freestream conditions
        self.freestream = Freestream(u_inf=1.0, alpha=self.alpha)

        A = build_singularity_matrix(A_source, B_vortex)
        b = build_freestream_rhs(self.panels, self.freestream)

        # solve for singularity strengths
        strengths = np.linalg.solve(A, b)

        # store source strength on each panel
        for i, panel in enumerate(self.panels):
            panel.sigma = strengths[i]

        # store circulation density
        gamma = strengths[-1]

        # tangential velocity at each panel center.
        compute_tangential_velocity(
            self.panels, self.freestream, gamma, A_source, B_vortex)

        # surface pressure coefficient
        compute_pressure_coefficient(self.panels, self.freestream)

        # !!! Without wake panels, Cl and Cd cannot be trusted apparently
        # https://www.symscape.com/blog/why_use_panel_method
        # Compute normal and tangent force coefficients, then inviscid cl and cd
        # cn = sum([-pnl.cp * pnl.length * pnl.sin_beta for pnl in self.panels])
        # ct = sum([-pnl.cp * pnl.length * pnl.cos_beta for pnl in self.panels])
        # cl = cn * self.freestream.cos_alpha - ct * self.freestream.sin_alpha
        # cd = cn * self.freestream.sin_alpha + ct * self.freestream.cos_alpha

        # Finish up
        self._user_readable = True

        return

    def show_Cp(self):
        """Plot surface pressure coefficient."""
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, dpi=140)

        ax.grid()
        ax.set_xlabel('$x/c$', fontsize=16)
        ax.set_ylabel('$C_p$', fontsize=16)
        ax.plot([panel.xc for panel in self.panels if panel.loc == 'upper'],
                [panel.cp for panel in self.panels if panel.loc == 'upper'],
                label='upper surface',
                color='r', linestyle='-', linewidth=2, marker='o',
                markersize=6)
        ax.plot([panel.xc for panel in self.panels if panel.loc == 'lower'],
                [panel.cp for panel in self.panels if panel.loc == 'lower'],
                label='lower surface',
                color='b', linestyle='-', linewidth=1, marker='o',
                markersize=6)

        ax.legend(loc='best', prop={'size': 16})
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(None, 1.5)
        ax.set_ylim(*ax.get_ylim()[::-1])
        ax.set_title(f'Number of panels: {self.panels.size}', fontsize=16)
        plt.show()
        return
