"""A module of various methods used to estimate wing aerodynamic performance."""
import numpy as np
from scipy.integrate import simpson, trapezoid

from carpy.aerodynamics.aerofoil import ThinAerofoil
from carpy.utility import Hint, Quantity, cast2numpy

__all__ = ["PLLT", "HorseshoeVortex"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Support Functions and Classes
# ---------------------------------------------------------------------------- #


def designate_sections(sections, mirror: bool = None, N: int = None) -> list:
    """
    Given a WingSections object, distribute sections over N-elements.

    Args:
        sections (WingSections): WingSections object, describing wing geometry.
        mirror: If set to True, the given sections should be mirrored to
            produce a symmetrical wing. Optional, defaults to True.
        N: The number of sections to return. Optional, defaults to 40.

    Returns:
        list: A list containing discretised WingSection elements.

    """
    # Recast as necessary
    mirror = True if mirror is None else False
    N = 40 if N is None else int(N)

    # Linearly spaced sections in defined span
    if mirror is True:
        if N % 2 == 0:  # Even number of horseshoes, don't define @y=0
            sections2rtn = sections[::N][1::2]
            sections2rtn = sections2rtn[::-1] + sections2rtn
        else:  # Odd number of horseshoes, definition @y=0 is required
            sections2rtn = sections[::int(np.ceil(N / 2))]  # ceil includes y=0
            sections2rtn = sections2rtn[:0:-1] + sections2rtn
    else:
        sections2rtn = sections[::N]

    return sections2rtn


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


class WingSolution(object):
    """
    Template object, of which solvers should aim to produce all attributes of.
    """
    _AR: float
    _CL: float
    _CLalpha: float
    _Sref: Quantity
    _b: Quantity
    _e: float
    _delta: float
    _tau: float

    def __str__(self):
        params2print = ["AR", "CL", "CLalpha", "Sref", "b", "e", "delta", "tau"]

        return_string = f"{type(self).__name__}\n"
        for param in params2print:
            if hasattr(self, f"_{param}"):
                return_string += f">\t{param:<8} = {getattr(self, param)}\n"

        return return_string

    @property
    def AR(self) -> float:
        """Wing aspect ratio, AR."""
        return self._AR

    @property
    def CL(self) -> float:
        """Wing lift coefficient, CL."""
        return self._CL

    @property
    def CLalpha(self) -> float:
        """Finite wing lift-curve slope, CLalpha."""
        return self._CLalpha

    @property
    def Sref(self) -> Quantity:
        """Wing reference planform area, Sref."""
        return self._Sref

    @property
    def b(self) -> Quantity:
        """Tip-to-tip span of the wing, b."""
        return self._b

    @property
    def e(self) -> float:
        """Planform span efficiency factor, e."""
        return self._e

    @property
    def delta(self) -> float:
        """Elliptical lift distribution deviation factor, delta."""
        return self._delta

    @property
    def tau(self) -> float:
        """Finite wing lift-curve slope deviation factor, tau."""
        return self._tau


# ============================================================================ #
# Public classes
# ---------------------------------------------------------------------------- #

class PLLT(WingSolution):
    """
    Prandtl's lifting line (a.k.a Lanchester-Prandtl wing) theory.

    For use in incompressible, inviscid, steady flow regimes with moderate to
    high aspect ratio, unswept wings.
    """

    def __init__(
            self, sections, span: Hint.num, alpha: Hint.num, N: int = None):
        # Recast as necessary
        N = 50 if N is None else int(N)

        # Create a cosine-spaced distribution of the given extent of sections
        section_max, section_min = max(sections), min(sections)

        # Skip first element to prevent the generation of singular matrices
        theta0 = np.linspace(0, np.pi, N + 2)[1:-1]

        # Station parameters (geometric)
        y = np.cos(theta0) * (span / 2)  # ... Span position
        Nsections = sections[
            np.interp(abs(y), [0, span / 2], [section_min, section_max])
        ]
        chord = [Nsection.chord for Nsection in Nsections]
        # Station parameters (aerodynamic)
        alpha_zl, Clalpha = [], []
        for i, Nsection in enumerate(Nsections):
            solution = ThinAerofoil(
                aerofoil=Nsection.aerofoil,
                alpha=alpha + Nsection.twist  # Effective AoA
            )
            alpha_zl.append(solution.alpha_zl - Nsection.twist)  # Effective ZL
            Clalpha.append(solution.Clalpha)

        chord, alpha_zl, Clalpha = map(cast2numpy, (chord, alpha_zl, Clalpha))

        # Solve matrices for Fourier series coefficients
        matA = np.zeros((N, N))
        for j in (n := np.arange(N)):
            term1 = 4 * span / Clalpha[j] / chord[j]
            term2 = (n + 1) / np.sin(theta0[j])
            matA[j] += np.sin((n + 1) * theta0[j]) * (term1 + term2)
        matB = (alpha - alpha_zl)[:, None]
        matX = np.linalg.solve(matA, matB)[:, 0]  # <-- This is the slow step!!!

        # Just some requisite quick maths :)
        wingarea = -simpson(y=chord, x=y)
        a0 = 2 * np.pi  # Ideal lift-curve slope
        n = (np.arange(1, len(matX) + 1))
        delta = (n * (matX / matX[0]) ** 2)[1:].sum()
        # n increases row-wise, theta0 is periodic column-wise
        # Take -ve of simpson integral because x=y moves from y=+b/2 to y=-b/2
        integrals = -simpson(y=chord * np.sin(n[:, None] * theta0), x=y)
        tau = span / (2 * wingarea) * (n * matX * integrals)[1:].sum()

        self._AR = span ** 2 / wingarea
        self._CL = np.pi * self.AR * matX[0]
        self._CLalpha = a0 / (1 + a0 / (np.pi * self.AR) * (1 + tau))
        self._Sref = Quantity(wingarea, "m^{2}")
        self._b = Quantity(span, "m")
        self._e = 1 / (1 + delta)
        self._delta = delta
        self._tau = tau

        return


class HorseshoeVortex(WingSolution):
    """
    Horseshoe Vortex Method over N-elements.

    For use in incompressible, inviscid, steady flow regimes.

    Notes:
        Currently assumes all lifting sections have a lift-slope of 2 pi.

    """

    def __init__(self, sections, span: Hint.num, alpha: Hint.num = None,
                 N: int = None, mirror: bool = None):
        """
        Args:
            sections: Wing sections object, defining 3D geometry.
            span: Tip-to-tip span of wing (if mirrored), else root-to-tip span.
            alpha: Angle of attack at the wing root (station zero).
            N: The number of horseshoe vortices. Optional, defaults to 40.
            mirror: Whether or not to treat the wing sections object as
                symmetrical around the smallest station index. Optional,
                defaults to True (mirrored wing).

        Notes:
            The reference coordinate system used in this code adheres to the
            NorthEastDown (NED) standard aircraft principle axes representation.

        """
        # Recast as necessary
        alpha = 0.0 if alpha is None else float(alpha)
        mirror = True if mirror is None else mirror
        N = 40 if N is None else int(N)

        # Problem setup
        large = 1e6
        Q = np.array([-np.cos(alpha), 0.0, -np.sin(alpha)])  # Freestream orient
        scalefactor = np.sqrt(N / (N + 1))  # Scaling fix for N = 1 case
        # Linearly spaced sections in defined span
        Nsections = designate_sections(sections=sections, mirror=mirror, N=N)
        # bprime = combined span of all horseshoe elements
        bprime = span * scalefactor
        # cprime = longitudinal dist. between control point and horseshoe front
        chord = np.array([section.chord for section in Nsections])
        cprime = np.array([[-0.5 * cw / scalefactor, 0, 0] for cw in chord])

        # Locate vortex segments (a,b), control vectors (c), and normal vectors
        va, vb, vc, n = np.zeros((4, N, 3))  # Initial assignment to 3D zeros.
        # ... distribute vortex locators (va, vb) in the y-direction
        if mirror is True:
            ys = np.linspace(-bprime / 2, bprime / 2, N + 1)
        else:
            ys = np.linspace(0, bprime, N + 1)
        va[:, 1] = ys[:-1]
        vb[:, 1] = ys[1:]
        # ... prescribe normal vectors as if wing is a flat plate
        n[:, 2] = -1.0

        # Define horseshoe geometry in the y >= 0, +ve direction
        dy = bprime / N
        for i in range(N):

            solution = ThinAerofoil(
                aerofoil=Nsections[i].aerofoil,
                alpha=alpha + Nsections[i].twist  # Effective AoA
            )
            alpha_zl = solution.alpha_zl - Nsections[i].twist  # Effective ZL

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

            # Starboard vortices
            else:
                # Sweep vortex locators (Move x-coordinates)
                tan_sweep = np.tan(Nsections[i].sweep)
                va[i + 1:, 0] -= dy * tan_sweep  # Outboard sections
                vb[i + 1:, 0] -= dy * tan_sweep
                va[i, 0] -= (dy / 2) * tan_sweep  # Current section
                vb[i, 0] -= (dy / 2) * tan_sweep
                del tan_sweep

                # Apply dihedral to vortex locators (Move z-coordinates)
                tan_dihedral = np.tan(Nsections[i].dihedral)
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
                cos_gamma = np.cos(-Nsections[i].dihedral)
                sin_gamma = np.sin(-Nsections[i].dihedral)
                rot_dihedral = np.array([
                    [1, 0, 0],  # About the X-axis
                    [0, cos_gamma, -sin_gamma],
                    [0, sin_gamma, cos_gamma]
                ])
                del cos_gamma, sin_gamma

                # Apply rotations to normal vector and control point locator
                n[i] = rot_dihedral @ (rot_twist @ n[i])
                vc[i] = (va[i] + vb[i]) / 2 + (rot_twist @ cprime[i])

        # Now if necessary (wing mirrored), copy starboard geometry to port side
        else:
            if mirror is True:
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
                infl += vfil(va[j] + np.array([-large, 0, 0]), va[j], vc[i])
                infl += vfil(vb[j], vb[j] + np.array([-large, 0, 0]), vc[i])
                A[i, j] = (infl * n).sum()
            rhs[i] = -(Q * n).sum()

        # Sum of circulation in z-direction should match freestream, given gamma
        gamma = np.linalg.solve(A, rhs)

        # Forces at centres of bound vortices (parallel with leading edge)
        Fx, Fy, Fz = np.zeros((3, N))
        bc = 0.5 * (va + vb)
        for i in range(N):
            # Local velocity vector on each load element i
            u = np.copy(Q)
            for j in range(N):
                u += vfil(va[j], vb[j], bc[i]) * gamma[j]
                u += vfil(
                    va[j] + np.array([-large, 0, 0]), va[j], bc[i]
                ) * gamma[j]
                u += vfil(
                    vb[j], vb[j] + np.array([-large, 0, 0]), bc[i]
                ) * gamma[j]
            # u cross s gives direction of action of the force from circulation
            s = vb[i] - va[i]
            Fx[i], Fy[i], Fz[i] = np.cross(u, s) * gamma[i]

        # Resolve these forces into perpendicular and parallel to freestream
        wingarea = trapezoid(sections[::(elems := 1000)].chord, dx=span / elems)
        halfS = wingarea / 2
        self._sectionCl = (Fx * np.sin(alpha) - Fz * np.cos(alpha)) / halfS
        self._sectionCdi = (-Fx * np.cos(alpha) - Fz * np.sin(alpha)) / halfS

        self._AR = span ** 2 / wingarea
        self._CL = self._sectionCl.sum()
        self._CLalpha = NotImplemented
        self._Sref = Quantity(wingarea, "m^{2}")
        self._b = Quantity(span, "m")
        self._e = self._CL ** 2 / np.pi / self._AR / np.sum(self._sectionCdi)
        self._delta = 1 / self._e - 1
        self._tau: float = NotImplemented

        return

    @property
    def sectionCL(self) -> np.ndarray:
        """
        An array describing the distribution of generated CL. This is from
        port to starboard in a full wingplane, or from an inboard section to the
        outboard sections.

        Returns:
            Section-wise distribution of CL components.

        """
        return self._sectionCl
