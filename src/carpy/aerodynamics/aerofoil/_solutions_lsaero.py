"""
Low-speed aerodynamics, many panel methods.

References:
        Katz, J. and Plotkin, A., "Low-Speed Aerodynamics: From Wing Theory to
            Panel Methods", McGraw-Hill inc., 1991, pp.302-369.
"""
import numpy as np
from scipy.integrate import simpson

from carpy.utility import Hint
from ._solutions import AerofoilSolution

__author__ = "Yaseen Reza"


# ============================================================================ #
# Support Functions and Classes
# ---------------------------------------------------------------------------- #

def VOR2D(Gammaj: Hint.num, x: Hint.num, z: Hint.num, xj: Hint.num,
          zj: Hint.num) -> tuple:
    """
    Computes the velocity components (u, w) at (x, z) due to a discrete vortex
    element of circulation strength Gammaj, located at (xj, zj).

    Args:
        Gammaj: Circulation strength Gamma of the vortex element.
        x: Location of a point to compute the velocity at.
        z: Location of a point to compute the velocity at.
        xj: Location of the vortex element.
        zj: Location of the vortex element.

    Returns:
        Velocities (u, w) at point (x, z).

    """
    rj_sq = (x - xj) ** 2 + (z - zj) ** 2

    factor = Gammaj / 2 / np.pi / rj_sq

    u = factor * (z - zj)
    w = factor * (xj - x)

    return u, w


def SORC2D(sigmaj: Hint.num, x: Hint.num, z: Hint.num, xj: Hint.num,
           zj: Hint.num) -> tuple:
    """
    Computes the velocity components (u, w) at (x, z) due to a discrete source
    element of circulation strength sigmaj, located at (xj, zj).

    Args:
        sigmaj: Source strength sigma of the source element.
        x: Location of a point to compute the velocity at.
        z: Location of a point to compute the velocity at.
        xj: Location of the source element.
        zj: Location of the source element.

    Returns:
        Velocities (u, w) at point (x, z).

    """
    rj_sq = (x - xj) ** 2 + (z - zj) ** 2

    factor = sigmaj / 2 / np.pi / rj_sq

    u = factor * (x - xj)
    w = factor * (z - zj)

    return u, w


# ============================================================================ #
# Public (solution) classes
# ---------------------------------------------------------------------------- #

class DiscreteVortexMethod(AerofoilSolution):
    """
    For thin cambered aerofoil problems.
    """

    def __init__(self, aerofoil, alpha: Hint.num, Npanels: int = None):
        # Super class call
        super().__init__(aerofoil, alpha, Npanels)

        # Obtain camber coordinates
        panel_dx = 1 / self._Npanels
        xcamb, ycamb = self._aerofoil._camber_points(step_target=panel_dx).T

        # Compute panel normals
        deta_dx = np.diff(ycamb) / np.diff(xcamb)
        panel_ns = (np.vstack([-deta_dx, np.ones(self._Npanels)]).T
                    / ((deta_dx[:, None] ** 2 + 1) ** 0.5))

        # Locate vortices
        xjs = xcamb[:-1] + panel_dx * (1 / 4)
        zjs = np.interp(xjs, xcamb, ycamb)

        # Locate collocation points
        xis = xcamb[:-1] + panel_dx * (3 / 4)
        zis = np.interp(xis, xcamb, ycamb)

        # Construct RHS
        Q = np.array([np.cos(alpha), np.sin(alpha)])
        RHS = np.sum(-Q * panel_ns, axis=1)

        # Spawn an influence coefficient array (each row == one collocation pt.)
        A = np.zeros((self._Npanels, self._Npanels))

        for i in range(self._Npanels):  # Loop over collocation points
            for j in range(self._Npanels):  # Loop over vortex points
                velocity_uw = VOR2D(
                    Gammaj=1.0,
                    x=xis[i], z=zis[i],
                    xj=xjs[j], zj=zjs[j]
                )
                A[i, j] = (velocity_uw * panel_ns[i]).sum()

        # Solve for point vortex strengths
        Gammas = np.linalg.solve(A, RHS)

        # Resolve lift and drag force per panel
        rho = 1
        dL, dD = (Gammas * rho * Q[:, None])

        # Resolve pressure force per panel
        V = 1.0  # Easily recognisable to be the case, magnitude of Q is unity
        panelsizes = (np.diff(xcamb) ** 2 + np.diff(ycamb) ** 2) ** 0.5
        dp = rho * V * Gammas / panelsizes

        # Compute centre of pressure
        if dp.sum() == 0:  # No pressure despite flow = zero lift!
            self._x_cp = np.inf  # Centre of pressure is infinitely behind LE
        else:
            self._x_cp = simpson(xjs * dp, x=xjs) / simpson(dp, x=xjs)

        # Resolve moment about the leading edge
        M0 = (dL * xjs * Q[0]).sum()

        # Non-dimensional coefficients
        c = 1.0
        self._xjs = xjs
        self._CL = dL.sum() / (0.5 * rho * V ** 2 * c)
        self._CD = dD.sum() / (0.5 * rho * V ** 2 * c)
        self._Cp = dp / (0.5 * rho * V ** 2)
        self._Cm_0 = M0 / (0.5 * rho * V ** 2 * c ** 2)

        return


class DiscreteSourceMethod(AerofoilSolution):
    """
    For thin symmetric aerofoil problems.
    """

    def __init__(self, aerofoil, alpha: Hint.num, Npanels: int = None):
        # Super class call
        super().__init__(aerofoil, alpha, Npanels)

        # Obtain aerofoil surface coordinates
        i_le = np.argmax(self._aerofoil._curvature)
        xupper = np.linspace(0, 1, self._Npanels + 1)
        yupper = np.interp(xupper, *self._aerofoil._points[:i_le][::-1].T)

        # Compute panel normals and tangents
        deta_dx = np.diff(yupper) / np.diff(xupper)
        panel_ns = (np.vstack([-deta_dx, np.ones(self._Npanels)]).T
                    / ((deta_dx[:, None] ** 2 + 1) ** 0.5))
        panel_ts = (np.array([[0, 1], [-1, 0]]) @ panel_ns.T).T  # rotate norm

        # Locate sources
        panel_dx = 1 / self._Npanels
        xjs = xupper[:-1] + panel_dx * (1 / 2)
        zjs = np.zeros(self._Npanels)

        # Locate collocation points
        xis = xjs.copy()
        zis = np.interp(xis, xupper, yupper)

        # Construct RHS
        Q = np.array([np.cos(alpha), np.sin(alpha)])
        RHS = np.sum(-Q * panel_ns, axis=1)

        # Spawn an influence coefficient array (each row == one collocation pt.)
        A = np.zeros((self._Npanels, self._Npanels))

        for i in range(self._Npanels):  # Loop over collocation points
            for j in range(self._Npanels):  # Loop over vortex points
                velocity_uw = SORC2D(
                    sigmaj=1.0,
                    x=xis[i], z=zis[i],
                    xj=xjs[j], zj=zjs[j]
                )
                A[i, j] = (velocity_uw * panel_ns[i]).sum()

        # Solve for point source strengths
        sigmas = np.linalg.solve(A, RHS)

        # Resolve velocities tangent to the panels
        Qtis = np.zeros_like(panel_ts)
        for i in range(xis.size):  # Loop over collocation points
            u, w = np.sum(SORC2D(sigmas, xis[i], zis[i], xjs, zjs), axis=1)
            Qtis[i] = ((u, w) + Q) * panel_ts[i]

        return
