"""
Low-speed aerodynamics, many panel methods.

References:
        Katz, J. and Plotkin, A., "Low-Speed Aerodynamics: From Wing Theory to
            Panel Methods", McGraw-Hill inc., 1991, pp.302-369.
"""
import numpy as np
from scipy.integrate import simpson

from carpy.geometry import Aerofoil
from carpy.utility import cast2numpy, moving_average
from ._common import AeroSolution

__all__ = ["PotentialFlow2D"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Support Functions and Classes
# ---------------------------------------------------------------------------- #

class PotentialFlow2D(object):
    """
    A collection of methods for computing the induced velocities of various
    potential flow elements (in 2D).
    """

    @staticmethod
    def source_D(sigmaj, x, z, xj, zj) -> tuple:
        """
        Computes the velocity components (u, w) at (x, z) due to a discrete
        source element of circulation strength sigmaj, located at (xj, zj).

        Args:
            sigmaj: Source strength sigma of the source element.
            x: Location of a point to compute the velocity at.
            z: Location of a point to compute the velocity at.
            xj: Location of the source element.
            zj: Location of the source element.

        Returns:
            Induced velocities (u, w) at point (x, z).

        """
        rj_sq = (x - xj) ** 2 + (z - zj) ** 2

        factor = sigmaj / 2 / np.pi / rj_sq

        u = factor * (x - xj)
        w = factor * (z - zj)
        if isinstance(u, np.ndarray) and u.size == 1:
            u = u[0]
            w = w[0]

        return u, w

    @staticmethod
    def doublet_D(muj, x, z, xj, zj, beta=None) -> tuple:
        """
        Computes the velocity components (u, w) at (x, z) due to a discrete
        doublet element of circulation strength muj, located at (xj, zj).

        Args:
            muj: Strength mu of the doublet element.
            x: Location of a point to compute the velocity at.
            z: Location of a point to compute the velocity at.
            xj: Location of the doublet element.
            zj: Location of the doublet element.
            beta: Orientation of element. Optional, defaults to 0.

        Returns:
            Induced velocities (u, w) at point (x, z).

        """
        # Recast as necessary
        x = cast2numpy(x)
        z = cast2numpy(z)
        beta = 0.0 if beta is None else beta

        dx, dz = x - xj, z - zj
        rj_sq = dx ** 2 + dz ** 2

        factor = muj / 2 / np.pi / (rj_sq ** 2)

        u = factor * -1 * (dx ** 2 - dz ** 2)
        w = factor * -2 * dx * dz

        # Apply rotation
        rot_matrix = np.array([
            [np.cos(beta), np.sin(beta)],
            [-np.sin(beta), np.cos(beta)]
        ])
        u, w = (rot_matrix @ np.vstack([u.flat, w.flat])).reshape((2, *u.shape))
        if u.size == 1:
            u = u[0]
            w = w[0]

        return u, w

    @staticmethod
    def vortex_D(Gammaj, x, z, xj, zj) -> tuple:
        """
        Computes the velocity components (u, w) at (x, z) due to a discrete
        vortex element of circulation strength Gammaj, located at (xj, zj).

        Args:
            Gammaj: Circulation strength Gamma of the vortex element.
            x: Location of a point to compute the velocity at.
            z: Location of a point to compute the velocity at.
            xj: Location of the vortex element.
            zj: Location of the vortex element.

        Returns:
            Induced velocities (u, w) at point (x, z).

        """
        rj_sq = (x - xj) ** 2 + (z - zj) ** 2

        factor = Gammaj / 2 / np.pi / rj_sq

        u = factor * (z - zj)
        w = factor * (xj - x)
        if isinstance(u, np.ndarray) and u.size == 1:
            u = u[0]
            w = w[0]

        return u, w

    @staticmethod
    def source_C(sigmaj, x, z, xj0, zj0, xj1, zj1) -> tuple:
        """
        Computes the velocity components (u, w) at (x, z) due to a panel source
        element of constant circulation strength sigmaj, with end points of the
        panel located by (xj0, zj0) and (xj1, zj1).

        Args:
            sigmaj: Source strength sigma of the source element.
            x: Location of a point to compute the velocity at.
            z: Location of a point to compute the velocity at.
            xj0: Location of endpoint 0 of a source panel.
            zj0: Location of endpoint 0 of a source panel.
            xj1: Location of endpoint 1 of a source panel.
            zj1: Location of endpoint 1 of a source panel.

        Returns:
            Induced velocities (u, w) at point (x, z).

        """
        # Recast as necessary
        x = cast2numpy(x)
        z = cast2numpy(z)

        # Locate coordinates
        xz = np.vstack([x.flat, z.flat])
        xz_panel = np.vstack([np.hstack([xj0, xj1]), np.hstack([zj0, zj1])])

        # Find the angle of the panel's coordinate system w.r.t global system
        # panel orientation angle, as defined by Katz and Plotkin
        alpha = float(np.pi - np.arctan2(*np.diff(xz_panel)[::-1]))
        rot_panel2global = np.array([
            [np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        rot_global2panel = np.linalg.inv(rot_panel2global)

        # Locate coordinates in the panel reference system
        origin = xz_panel[:, 1][:, None]  # place origin at (xj1, zj1)
        xz_p = rot_global2panel @ (xz - origin)
        xz_p_panel = rot_global2panel @ (xz_panel - origin)

        # Compute induced velocities using the panel reference system
        factor = sigmaj / 2 / np.pi
        r1_sq = (xz_p[0] - xz_p_panel[0][1]) ** 2 + xz_p[1] ** 2
        r2_sq = (xz_p[0] - xz_p_panel[0][0]) ** 2 + xz_p[1] ** 2
        u_p = 0.5 * factor * np.log(r1_sq / r2_sq)
        theta1 = np.arctan2(xz_p[1], (xz_p[0] - xz_p_panel[0][1]))
        theta2 = np.arctan2(xz_p[1], (xz_p[0] - xz_p_panel[0][0]))
        w_p = factor * (theta2 - theta1)

        # Transform induced velocities back into the global frame
        u, w = (rot_panel2global @ np.vstack([u_p, w_p])).reshape((2, *x.shape))
        if u.size == 1:
            u = u[0]
            w = w[0]

        return u, w

    @staticmethod
    def doublet_C(muj, x, z, xj0, zj0, xj1, zj1) -> tuple:
        """
        Computes the velocity components (u, w) at (x, z) due to a panel doublet
        element of constant strength muj, with end points of the
        panel located by (xj0, zj0) and (xj1, zj1).

        Args:
            muj: Strength mu of the doublet element.
            x: Location of a point to compute the velocity at.
            z: Location of a point to compute the velocity at.
            xj0: Location of endpoint 0 of a source panel.
            zj0: Location of endpoint 0 of a source panel.
            xj1: Location of endpoint 1 of a source panel.
            zj1: Location of endpoint 1 of a source panel.

        Returns:
            Induced velocities (u, w) at point (x, z).

        """
        # Recast as necessary
        x = cast2numpy(x)
        z = cast2numpy(z)

        # Locate coordinates
        xz = np.vstack([x.flat, z.flat])
        xz_panel = np.vstack([np.hstack([xj0, xj1]), np.hstack([zj0, zj1])])

        # Find the angle of the panel's coordinate system w.r.t global system
        # panel orientation angle, as defined by Katz and Plotkin
        alpha = float(np.pi - np.arctan2(*np.diff(xz_panel)[::-1]))
        rot_panel2global = np.array([
            [np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        rot_global2panel = np.linalg.inv(rot_panel2global)

        # Locate coordinates in the panel reference system
        origin = xz_panel[:, 1][:, None]  # place origin at (xj1, zj1)
        xz_p = rot_global2panel @ (xz - origin)
        xz_p_panel = rot_global2panel @ (xz_panel - origin)

        # Compute induced velocities using the panel reference system
        factor = muj / 2 / np.pi
        r1_sq = (xz_p[0] - xz_p_panel[0][1]) ** 2 + xz_p[1] ** 2
        r2_sq = (xz_p[0] - xz_p_panel[0][0]) ** 2 + xz_p[1] ** 2
        u_p = factor * xz_p[1] * (1 / r1_sq - 1 / r2_sq)
        term1 = (xz_p[0] - xz_p_panel[0][1]) / r1_sq
        term2 = (xz_p[0] - xz_p_panel[0][0]) / r2_sq
        w_p = factor * -1 * (term1 - term2)

        # Transform induced velocities back into the global frame
        u, w = (rot_panel2global @ np.vstack([u_p, w_p])).reshape((2, *x.shape))
        if u.size == 1:
            u = u[0]
            w = w[0]

        return u, w

    @staticmethod
    def vortex_C(gammaj, x, z, xj0, zj0, xj1, zj1) -> tuple:
        """
        Computes the velocity components (u, w) at (x, z) due to a panel vortex
        element of constant strength gammaj, with end points of the
        panel located by (xj0, zj0) and (xj1, zj1).

        Args:
            gammaj: Vortex strength gamma of the vortex element.
            x: Location of a point to compute the velocity at.
            z: Location of a point to compute the velocity at.
            xj0: Location of endpoint 0 of a source panel.
            zj0: Location of endpoint 0 of a source panel.
            xj1: Location of endpoint 1 of a source panel.
            zj1: Location of endpoint 1 of a source panel.

        Returns:
            Induced velocities (u, w) at point (x, z).

        """
        # Recast as necessary
        x = cast2numpy(x)
        z = cast2numpy(z)

        # Locate coordinates
        xz = np.vstack([x.flat, z.flat])
        xz_panel = np.vstack([np.hstack([xj0, xj1]), np.hstack([zj0, zj1])])

        # Find the angle of the panel's coordinate system w.r.t global system
        # panel orientation angle, as defined by Katz and Plotkin
        alpha = float(np.pi - np.arctan2(*np.diff(xz_panel)[::-1]))
        rot_panel2global = np.array([
            [np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        rot_global2panel = np.linalg.inv(rot_panel2global)

        # Locate coordinates in the panel reference system
        origin = xz_panel[:, 1][:, None]  # place origin at (xj1, zj1)
        xz_p = rot_global2panel @ (xz - origin)
        xz_p_panel = rot_global2panel @ (xz_panel - origin)

        # Compute induced velocities using the panel reference system
        factor = gammaj / 2 / np.pi
        dx1, dx2 = xz_p[0] - xz_p_panel[0][1], xz_p[0] - xz_p_panel[0][0]
        dz1, dz2 = xz_p[1] - xz_p_panel[1][1], xz_p[1] - xz_p_panel[1][0]
        u_p = factor * (np.arctan2(dz2, dx2) - np.arctan2(dz1, dx1))
        w_p = factor * -0.5 * np.log(
            (dx1 ** 2 + dz1 ** 2) / (dx2 ** 2 + dz2 ** 2))

        # Transform induced velocities back into the global frame
        u, w = (rot_panel2global @ np.vstack([u_p, w_p])).reshape((2, *x.shape))
        if u.size == 1:
            u = u[0]
            w = w[0]

        return u, w


# ============================================================================ #
# Public (solution) classes
# ---------------------------------------------------------------------------- #

class DiscreteVortexMethod(AeroSolution):
    """
    Numerical solution for thin cambered aerofoil problems.
    """

    def __init__(self, aerofoil: Aerofoil, Npanels: int = None, **kwargs):
        # Super class call
        if Npanels is None:  # Can't add 1 to a None!
            super().__init__(aerofoil, **kwargs)
        else:
            super().__init__(aerofoil, N=Npanels + 1, **kwargs)

        # Obtain camber coordinates
        panel_dx = 1 / (Npanels := self._Nctrlpts - 1)
        xcamb, ycamb = self.sections._camber_points(step_target=panel_dx).T

        # Compute panel normals
        deta_dx = np.diff(ycamb) / np.diff(xcamb)
        panel_ns = np.vstack([-deta_dx, np.ones(Npanels)])  # vector > 1
        panel_ns = panel_ns / np.linalg.norm(panel_ns, axis=0)  # normalised

        # Locate vortices
        xjs = xcamb[:-1] + panel_dx * (1 / 4)
        zjs = np.interp(xjs, xcamb, ycamb)

        # Locate collocation points
        xis = xcamb[:-1] + panel_dx * (3 / 4)
        zis = np.interp(xis, xcamb, ycamb)

        # Construct RHS
        Q = np.array([np.cos(self.alpha), np.sin(self.alpha)])
        RHS = np.dot(-Q, panel_ns)

        # Spawn an influence coefficient array (each row == one collocation pt.)
        A = np.zeros((Npanels, Npanels))

        for i in range(Npanels):  # Loop over collocation points
            for j in range(Npanels):  # Loop over vortex points
                velocity_uw = PotentialFlow2D.vortex_D(
                    Gammaj=1.0,
                    x=xis[i], z=zis[i],
                    xj=xjs[j], zj=zjs[j]
                )
                A[i, j] = (velocity_uw * panel_ns[:, i]).sum(axis=0)

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

        # Finish up
        self._user_readable = True

        return


class DiscreteSourceMethod(AeroSolution):
    """
    Numerical solution for symmetric aerofoil problems.
    """

    def __init__(self, aerofoil: Aerofoil, Npanels: int = None, **kwargs):
        # Super class call
        if Npanels is None:  # Can't add 1 to a None!
            super().__init__(aerofoil, **kwargs)
        else:
            super().__init__(aerofoil, N=Npanels + 1, **kwargs)

        # Obtain aerofoil surface coordinates
        i_le = np.argmax(self.sections._curvature)
        xupper = np.linspace(0, 1, self._Nctrlpts)
        yupper = np.interp(xupper, *self.sections.points[:i_le][::-1].T)

        # Compute panel normals and tangents
        Npanels = self._Nctrlpts - 1
        deta_dx = np.diff(yupper) / np.diff(xupper)
        panel_ns = np.vstack([-deta_dx, np.ones(Npanels)])  # vector > 1
        panel_ns = panel_ns / np.linalg.norm(panel_ns, axis=0)  # normalised
        panel_ts = np.array([[0, 1], [-1, 0]]) @ panel_ns  # rotate norm

        # Locate sources
        panel_dx = 1 / Npanels
        xjs = xupper[:-1] + panel_dx * (1 / 2)
        zjs = np.zeros(Npanels)

        # Locate collocation points
        xis = xjs.copy()
        xis[0] -= panel_dx * (2 / 5)  # Translate first collocation towards LE
        xis[-1] += panel_dx * (2 / 5)  # Translate last collocation towards TE
        zis = np.interp(xis, xupper, yupper)

        # Construct RHS
        Q = np.array([np.cos(self.alpha), np.sin(self.alpha)])
        RHS = np.dot(-Q, panel_ns)

        # Spawn an influence coefficient array (each row == one collocation pt.)
        A = np.zeros((Npanels, Npanels))

        for i in range(Npanels):  # Loop over collocation points
            for j in range(Npanels):  # Loop over vortex points
                velocity_uw = PotentialFlow2D.source_D(
                    sigmaj=1.0,
                    x=xis[i], z=zis[i],
                    xj=xjs[j], zj=zjs[j]
                )
                A[i, j] = (velocity_uw * panel_ns[:, i]).sum(axis=0)

        # Solve for point source strengths
        sigmas = np.linalg.solve(A, RHS)

        if ~np.isclose(sigmas.sum(), 0.0, atol=1e-3):
            errormsg = f"sigmas.sum() != 0, bad result (got {sigmas.sum()=})"
            raise RuntimeError(errormsg)

        # Resolve velocities tangent to the panels
        Qtis = np.zeros_like(xis)
        for i in range(Qtis.size):  # Loop over collocation points
            u, w = np.sum(
                PotentialFlow2D.source_D(sigmas, xis[i], zis[i], xjs, zjs),
                axis=1
            )
            Qtis[i] = np.dot((u, w) + Q, panel_ts[:, i])

        self._xjs = xjs
        self._Cp = 1 - Qtis  # / Qs=1.0

        # Finish up
        self._user_readable = True

        return


class ConstantSourceMethod(AeroSolution):
    """
    Numerical solution for symmetric aerofoil problems.
    """

    def __init__(self, aerofoil: Aerofoil, Npanels: int = None, **kwargs):
        # Super class call
        if Npanels is None:  # Can't add 1 to a None!
            super().__init__(aerofoil, **kwargs)
        else:
            super().__init__(aerofoil, N=Npanels + 1, **kwargs)

        # Obtain aerofoil surface coordinates
        i_le = np.argmax(self.sections._curvature)
        xupper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, self._Nctrlpts)))
        yupper = np.interp(xupper, *self.sections.points[:i_le][::-1].T)

        # Compute panel normals and tangents
        Npanels = self._Nctrlpts - 1
        deta_dx = np.diff(yupper) / np.diff(xupper)
        panel_ns = np.vstack([-deta_dx, np.ones(Npanels)])  # vector > 1
        panel_ns = panel_ns / np.linalg.norm(panel_ns, axis=0)  # normalised
        # panel_ts = np.array([[0, 1], [-1, 0]]) @ panel_ns  # rotate norm

        # Locate sources
        xj0s, xj1s = xupper[::-1][:-1], xupper[::-1][1:]
        zj0s, zj1s = np.zeros((2, Npanels))

        # Locate collocation points
        xis = moving_average(x=xupper, w=2)
        zis = np.interp(xis, xupper, yupper)

        # Construct RHS
        Q = np.array([np.cos(self.alpha), np.sin(self.alpha)])
        RHS = np.dot(-Q, panel_ns)

        # Spawn an influence coefficient array (each row == one collocation pt.)
        A = np.zeros((Npanels, Npanels))

        for i in range(Npanels):  # Loop over collocation points
            for j in range(Npanels):  # Loop over vortex points

                # if i == j:  # Exclude self-influence, enforce flow tangency
                #     A[i, j] = 0.5  # Strength / 2 = freestream velocity
                #     continue

                velocity_uw = PotentialFlow2D.source_C(
                    sigmaj=1.0,
                    x=xis[i], z=zis[i],
                    xj0=xj0s[j], zj0=zj0s[j],
                    xj1=xj1s[j], zj1=zj1s[j]
                )
                A[i, j] = (velocity_uw * panel_ns[:, i]).sum(axis=0)

        # Solve for point source strengths
        sigmas = np.linalg.solve(A, RHS)

        if ~np.isclose(sigmas.sum(), 0.0, atol=1e-3):
            errormsg = f"sigmas.sum() != 0, bad result (got {sigmas.sum()=})"
            raise RuntimeError(errormsg)

        # Finish up
        self._user_readable = True

        return
