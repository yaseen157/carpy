"""A module of lifting line methods for predicting aerodynamic performance."""
import numpy as np

from carpy.geometry import WingSections
from carpy.utility import Hint, cast2numpy
from ._common import AeroSolution
from ._soln2d_thinaerofoils import ThinAerofoil2D

__all__ = ["PrandtlLLT", "HorseshoeVortex"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Support Functions and Classes
# ---------------------------------------------------------------------------- #


# noinspection PyUnusedLocal
def vfil(xyzA: Hint.nums, xyzB: Hint.nums, xyzC: Hint.nums):
    """
    Compute the influence of vortex line AB at control point C, return zero for
    small denominators (to exclude self-influence).

    Args:
        xyzA: Vector describing position of reference point A in 3D space.
        xyzB: Vector describing position of reference point B in 3D space.
        xyzC: Vector describing position of control point C in 3D space.

    Returns:
        Influence I of vortex line AB at control point C.

    """
    # Recast as necessary
    xyzA, xyzB, xyzC = cast2numpy([xyzA, xyzB, xyzC])

    small = 1e-12
    ab = xyzB - xyzA
    ac = xyzC - xyzA
    bc = xyzC - xyzB
    # noinspection PyUnreachableCode
    cross = np.cross(ac, bc)
    den = 4 * np.pi * (cross ** 2).sum()
    abs_ac = np.sqrt((ac ** 2).sum())
    abs_bc = np.sqrt((bc ** 2).sum())
    num = (ab * (ac / abs_ac - bc / abs_bc)).sum()

    if den <= small:
        influence = np.zeros(3)
    else:
        influence = num / den * cross

    return influence


class PrandtlLLT(AeroSolution):
    """
    Prandtl's lifting line (a.k.a. Lanchester-Prandtl wing) theory.

    For use in incompressible, inviscid, steady flow regimes with moderate to
    high aspect ratio, unswept wings.
    """

    def __init__(self, wingsections: WingSections, **kwargs):
        # Super class call
        super().__init__(wingsections, **kwargs)

        # Library limitations
        self._CY = 0  # This is an assumption, I'm not actually sure...
        if (beta := self.beta) != 0:
            raise ValueError(f"Sorry, non-zero beta is unsupported ({beta=})")

        # Make sure the wing is symmetric
        if not self.sections.mirrored:
            errormsg = f"{type(self).__name__} only works with symmetric wings"
            raise ValueError(errormsg)

        # Cosine distribution of wing sections (skip first and last section)
        theta0 = np.linspace(0, np.pi, (N := self._Nctrlpts) + 2)[1:-1]
        sections = wingsections.spansections(N=N + 2, dist_type="cosine")[1:-1]

        # Station: geometric parameters
        chord = [sec.chord.x for sec in sections]
        # Station: aerodynamic parameters
        alpha_zl, Clalpha = [], []
        for i, sec in enumerate(sections):
            aoa = self.alpha + sec.twist  # Effective angle of attack
            soln_thinaero = ThinAerofoil2D(aerofoil=sec.aerofoil, alpha=aoa)
            alpha_zl.append(soln_thinaero._alpha_zl - sec.twist)  # Eff zerolift
            Clalpha.append(soln_thinaero._Clalpha)

        chord, alpha_zl, Clalpha = map(cast2numpy, (chord, alpha_zl, Clalpha))

        # Solve matrices for Fourier series coefficients
        matA = np.zeros((N, N))
        for j in (n := np.arange(N)):
            term1 = 4 * wingsections.b.x / Clalpha[j] / chord[j]
            term2 = (n + 1) / np.sin(theta0[j])
            matA[j] += np.sin((n + 1) * theta0[j]) * (term1 + term2)
        matB = (self.alpha - alpha_zl)[:, None]
        matX = np.linalg.solve(matA, matB)[:, 0]  # <-- This is the slow step!!!

        # Comute induced drag
        delta = (np.arange(1, len(matX) + 1) * (matX / matX[0]) ** 2)[1:].sum()
        e = 1 / (1 + delta)

        # Assignments
        self._CL = np.pi * self.sections.AR * matX[0]
        self._CDi = self.CL ** 2 / (np.pi * e * wingsections.AR)
        self._CY = 0.0

        # Finish up
        self._user_readable = True

        return


class HorseshoeVortex(AeroSolution):
    """
    Horseshoe Vortex Method, with N vortices.

    For use in incompressible, inviscid, steady flow regimes.

    Notes:
        Currently assumes all sections obey thin aerofoil results.

    """

    def __init__(self, wingsections: WingSections, **kwargs):
        super().__init__(wingsections, **kwargs)

        # Problem setup
        # ... create rotation matrix for angle of attack and of sideslip
        cos_alpha, sin_alpha = np.cos(-(alpha := self.alpha)), np.sin(-alpha)
        rot_alpha = np.array([
            [cos_alpha, 0, sin_alpha],
            [0, 1, 0],  # About the Y-axis
            [-sin_alpha, 0, cos_alpha]
        ])
        cos_beta, sin_beta = np.cos(-self.beta), np.sin(-self.beta)
        rot_beta = np.array([
            [cos_beta, -sin_beta, 0],
            [sin_beta, cos_beta, 0],
            [0, 0, 1]  # About the Z-axis
        ])
        # ... orient the freestream vector
        Q = rot_beta @ rot_alpha @ np.array([-1.0, 0.0, 0.0])

        # ... scale factor fix for N = 1 case
        scalefactor = np.sqrt((N := self._Nctrlpts) / (N + 1))

        # ... linearly spaced sections of the wing
        sections = wingsections.spansections(N=N, dist_type="linear")

        # ... combined span of all the horseshoe elements
        bprime = wingsections.b.x * scalefactor

        # ... longitudinal distance between control point and aerodynamic centre
        chords = [sec.chord.x for sec in sections]
        cprime = np.array([[-0.5 * cw / scalefactor, 0, 0] for cw in chords])

        # Locate vortex segments (a,b), control vectors (c), and normal vectors
        va, vb, vc, n = np.zeros((4, N, 3))  # Initial assignment to 3D zeros.
        # ... distribute vortex locators (va, vb) in the y-direction
        if wingsections.mirrored is True:
            ys = np.linspace(-bprime / 2, bprime / 2, N + 1)
        else:
            ys = np.linspace(0, bprime, N + 1)
        va[:, 1] = ys[:-1]
        vb[:, 1] = ys[1:]
        # ... prescribe normal vectors as if wing is a flat plate
        n[:, 2] = -1.0

        # Locate semi-infinite vortices
        vinf = np.zeros((N, 3))
        vinf[:, 0] = -(large := 1e6)

        # Define horseshoe geometry in the y >= 0, +ve direction
        dy = bprime / N
        for i in range(N):

            solution = ThinAerofoil2D(
                aerofoil=sections[i].aerofoil,
                alpha=alpha + sections[i].twist  # Effective AoA
            )
            alpha_zl = solution._alpha_zl - sections[i].twist  # Effective ZL

            # Port-side vortices (any mirrored elements, if they exist)
            if va[i, 1] < 0 and vb[i, 1] <= 0:
                continue  # skip, we'll populate this later

            # Centreline vortex (vortex symmetrical about the centreline)
            elif va[i, 1] < 0:
                # Create rotation matrix for aerodynamic + geometric twist
                # section.alpha_zl automatically combines aero + geo twist...
                cos_theta = np.cos(-alpha_zl)
                sin_theta = np.sin(-alpha_zl)
                rot_twist = np.array([
                    [cos_theta, 0, sin_theta],
                    [0, 1, 0],  # About the Y-axis
                    [-sin_theta, 0, cos_theta]
                ])
                del cos_theta, sin_theta

                # Apply rotation to normal vector and control point locator
                n[i] = rot_twist @ n[i]
                vc[i] = (va[i] + vb[i]) / 2 + (rot_twist @ cprime[i])
                vinf[i] = rot_twist @ vinf[i]  # Twist semi-infinite vortices
                # Dihedral and sweep are safely ignored at the centreline

            # Starboard vortices
            else:
                # Sweep vortex locators (Move x-coordinates)
                # ... this assumes that the vortex locators (which are more
                # ... akin to centre of pressure locators) sweep by the same
                # ... amount that the leading edge does. Not true in real life!
                tan_sweep = np.tan(sections[i].sweep)
                va[i + 1:, 0] -= dy * tan_sweep  # Outboard sections
                vb[i + 1:, 0] -= dy * tan_sweep
                va[i, 0] -= (dy / 2) * tan_sweep  # Current section
                vb[i, 0] -= (dy / 2) * tan_sweep
                del tan_sweep

                # Apply dihedral to vortex locators (Move z-coordinates)
                tan_dihedral = np.tan(sections[i].dihedral)
                va[i + 1:, 2] -= dy * tan_dihedral  # Outboard sections
                vb[i + 1:, 2] -= dy * tan_dihedral
                va[i, 2] -= (dy / 2) * tan_dihedral  # Current section
                vb[i, 2] -= (dy / 2) * tan_dihedral
                del tan_dihedral

                # Create rotation matrix for aerodynamic + geometric twist
                # section.alpha_zl automatically combines aero + geo twist...
                cos_theta = np.cos(-alpha_zl)
                sin_theta = np.sin(-alpha_zl)
                rot_twist = np.array([
                    [cos_theta, 0, sin_theta],
                    [0, 1, 0],  # About the Y-axis
                    [-sin_theta, 0, cos_theta]
                ])
                del cos_theta, sin_theta

                # Create rotation matrix for dihedral
                # Take negatives, because matrix rotation assumes +ve==clockwise
                cos_gamma = np.cos(-sections[i].dihedral)
                sin_gamma = np.sin(-sections[i].dihedral)
                rot_dihedral = np.array([
                    [1, 0, 0],  # About the X-axis
                    [0, cos_gamma, -sin_gamma],
                    [0, sin_gamma, cos_gamma]
                ])
                del cos_gamma, sin_gamma

                # Apply rotations to normal vector and control point locator
                n[i] = rot_dihedral @ (rot_twist @ n[i])
                vc[i] = (va[i] + vb[i]) / 2 + (rot_twist @ cprime[i])
                # apparently, the below step is controversial???
                # vinf[i] = rot_twist @ vinf[i]  # Twist semi-infinite vortices

        # Now if necessary (wing mirrored), copy starboard geometry to port side
        else:
            if wingsections.mirrored is True:
                mirror_idx = int(N / 2)
                # For consistency, va and vb must swap in the mirror image
                va[:mirror_idx] = vb[-mirror_idx:][::-1] * np.array([1, -1, 1])
                vb[:mirror_idx] = va[-mirror_idx:][::-1] * np.array([1, -1, 1])
                # ", but mirroring vc
                vc[:mirror_idx] = vc[-mirror_idx:][::-1] * np.array([1, -1, 1])
                # Simple mirror for normal vectors
                n[:mirror_idx] = n[-mirror_idx:][::-1]

        # Set matrix and RHS elements, solve for circulations (Gamma)
        A = np.zeros((N, N))
        rhs = np.zeros((N, 1))

        for i in range(N):
            for j in range(N):
                # Influence contributions; 'LARGE -> va -> vb -> LARGE' on vc
                infl = vfil(va[j], vb[j], vc[i])
                infl += vfil(va[j] + vinf[j], va[j], vc[i])
                infl += vfil(vb[j], vb[j] + vinf[j], vc[i])
                A[i, j] = (infl * n).sum()
            rhs[i] = -(Q * n).sum()

        # Sum of circulation in z-direction should match freestream, given gamma
        gamma = np.linalg.solve(A, rhs)

        # Forces at centres of bound vortices (parallel with leading edge)
        Fxyz = np.zeros((3, N))
        self._xyz_cp = 0.5 * (va + vb)
        for i in range(N):
            # Local velocity vector on each load element i
            u = np.copy(Q)
            for j in range(N):
                u += vfil(va[j], vb[j], self._xyz_cp[i]) * gamma[j]
                u += vfil(va[j] + vinf[j], va[j], self._xyz_cp[i]) * gamma[j]
                u += vfil(vb[j], vb[j] + vinf[j], self._xyz_cp[i]) * gamma[j]
            # u cross s gives direction of action of the force from circulation
            s = vb[i] - va[i]
            # noinspection PyUnreachableCode
            Fxyz[:, i] = np.cross(u, s) * gamma[i]

        # Resolve these forces into perpendicular and parallel to freestream
        orient_mat = np.linalg.inv(rot_beta) @ np.linalg.inv(rot_alpha)
        self._sectionCDi, self._sectionCY, self._sectionCL = \
            -(orient_mat @ Fxyz / 0.5 / wingsections.Sref.x)

        # Make assignments
        self._CDi = self._sectionCDi.sum()
        self._CY = self._sectionCY.sum()
        self._CL = self._sectionCL.sum()

        # Finish up
        self._user_readable = True

        return
