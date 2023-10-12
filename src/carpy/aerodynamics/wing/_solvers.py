"""A module of various methods used to estimate wing aerodynamic performance."""
import numpy as np
from scipy.integrate import simpson, trapezoid

from carpy.aerodynamics.aerofoil import ThinAerofoil
from carpy.environment import ISA1975
from carpy.structures import DiscreteIndex
from carpy.utility import Hint, Quantity, cast2numpy, moving_average

__all__ = ["PLLT", "HorseshoeVortex", "CDfGudmundsson"]  # Cantilever1DStatic"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Support Functions and Classes
# ---------------------------------------------------------------------------- #


def designate_sections(sections: DiscreteIndex, mirror: bool = None,
                       N: int = None) -> list:
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
    mirror = True if mirror is None else mirror
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

    def __init__(self, sections: DiscreteIndex, span: Hint.num, alpha: Hint.num,
                 N: int = None):
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
        chord = [Nsection.chord.x for Nsection in Nsections]
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

    def __init__(self, sections: DiscreteIndex, span: Hint.num, alpha: Hint.num,
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
        N = 40 if N is None else int(N)
        mirror = True if mirror is None else mirror

        if mirror is False:
            raise NotImplementedError("Sorry, this method isn't ready yet.")

        # Problem setup
        large = 1e6
        Q = np.array([-np.cos(alpha), 0.0, -np.sin(alpha)])  # Freestream orient
        scalefactor = np.sqrt(N / (N + 1))  # Scaling fix for N = 1 case
        # Linearly spaced sections in defined span
        Nsections = designate_sections(sections=sections, mirror=mirror, N=N)
        self._Nsections = Nsections
        # bprime = combined span of all horseshoe elements
        bprime = span * scalefactor
        # cprime = longitudinal dist. between control point and horseshoe front
        chord = np.array([section.chord.x for section in Nsections])
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

        # Locate semi-infinite vortices
        vinf = np.zeros((N, 3))
        vinf[:, 0] = -large

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
                vinf[i] = rot_twist @ vinf[i]  # Twist semi-infinite vortices
                # Dihedral and sweep are safely ignored at the centreline

            # Starboard vortices
            else:
                # Sweep vortex locators (Move x-coordinates)
                # ... this assumes that the vortex locators (which are more
                # ... akin to centre of pressure locators) sweep by the same
                # ... amount that the leading edge does. Not true in real life!
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
                # apparently, the below step is controversial???
                vinf[i] = rot_twist @ vinf[i]  # Twist semi-infinite vortices

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
                infl += vfil(va[j] + vinf[j], va[j], vc[i])
                infl += vfil(vb[j], vb[j] + vinf[j], vc[i])
                A[i, j] = (infl * n).sum()
            rhs[i] = -(Q * n).sum()

        # Sum of circulation in z-direction should match freestream, given gamma
        gamma = np.linalg.solve(A, rhs)

        # Forces at centres of bound vortices (parallel with leading edge)
        Fx, Fy, Fz = np.zeros((3, N))
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
            Fx[i], Fy[i], Fz[i] = np.cross(u, s) * gamma[i]

        # Resolve these forces into perpendicular and parallel to freestream
        elems = 1000
        area_chords = [qty.x for qty in sections[::elems].chord]
        wingarea = trapezoid(area_chords, dx=span / elems)
        halfS = wingarea / 2
        self._Cl = (Fx * np.sin(alpha) - Fz * np.cos(alpha)) / halfS
        self._Cdi = (-Fx * np.cos(alpha) - Fz * np.sin(alpha)) / halfS

        self._AR = span ** 2 / wingarea
        self._CL = self._Cl.sum()
        self._CLalpha = NotImplemented
        self._Sref = Quantity(wingarea, "m^{2}")
        self._b = Quantity(span, "m")
        self._e = self._CL ** 2 / np.pi / self._AR / self._Cdi.sum()
        self._delta = 1 / self._e - 1
        self._tau: float = NotImplemented

        return

    @property
    def Cl(self) -> np.ndarray:
        """
        An array describing the distribution of generated CL. This is from
        port to starboard in a full wingplane, or from an inboard section to the
        outboard sections.

        Returns:
            Section-wise distribution of CL components.

        """
        return self._Cl

    @property
    def xyz_cp(self) -> np.ndarray:
        """
        An array of points, describing the distribution of wing section centres
        of pressures in physical space.

        Returns:
            Array of (N,3) points, describing xyz coordinates of sectional
            centres of pressure.

        """
        return self._xyz_cp


class Cantilever1DStatic(WingSolution):

    def __init__(self, sections, spar, span: Hint.num, alpha: Hint.num,
                 lift: Hint.num, N: int = None, mirror: bool = None,
                 model: str = None):
        # Define compatability parameters
        supported_models = [HorseshoeVortex]
        supported_models = dict(zip(
            [x.__name__ for x in supported_models],
            supported_models
        ))

        # Recast as necessary
        kwargs = {
            "sections": sections, "spar": spar, "span": span, "alpha": alpha,
            "lift": lift, "N": 40 if N is None else int(N),
            "mirror": True if mirror is None else mirror,
            "model": model
        }
        if kwargs["model"] is None:  # Model unspecified
            kwargs["model"] = HorseshoeVortex
        elif supported_models.get(model) is None:  # Model unsupported
            errormsg = f"{model=} is unsupported, try one of {supported_models}"
            raise ValueError(errormsg)

        # Evaluate aerodynamic model
        # noinspection PyUnresolvedReferences
        soln_aero = kwargs["model"](
            **{  # Iterate over kwargs, and pass only those which appear in init
                k: v for (k, v) in kwargs.items()
                if k in kwargs["model"].__init__.__annotations__  # <-Typehints!
            }
        )

        # Find out if solution is mirrored
        ys = soln_aero.xyz_cp[:, 1]
        if kwargs["mirror"] is True:
            root_i = int(np.ceil(len(ys) / 2))
            ys = ys[root_i:]
        else:
            root_i = 0

        # Lift and moment component distribution
        sectionL = lift * (soln_aero.Cl[root_i:] / soln_aero.CL)
        sectionM = sectionL * ys

        # Flexural modulus of carbon fibre (?)
        E_rect = 60e9  # e9 == GPa
        # Subscript x as it's the effect of material distributed about x-axis
        t_wall = 1e-3
        b_spar = 50e-3
        h0_spar = 100e-3
        h1_spar = 20e-3

        def Ix_rect(y):
            """Variable cross-section, hollow rectangular spar."""
            y = np.abs(y)
            h_spar = np.interp(y, [0, 10], [h0_spar, h1_spar])
            Ixx_o = b_spar * h_spar ** 3 / 12
            Ixx_i = (b_spar - 2 * t_wall) * (h_spar - 2 * t_wall) ** 3 / 12
            return Ixx_o - Ixx_i

        EI = E_rect * Ix_rect(ys)

        # Distribution of angular increments with each section
        sectiontheta = sectionM / EI

        # Distibution of vertical displacement with each section
        sectionnu = np.cumsum(sectiontheta)

        plot_dy, = np.diff(ys[0:2])
        plot_y = np.zeros_like(sectionnu)
        plot_z = np.copy(sectionnu)

        for i in range(1, len(sectiontheta)):
            plot_y[i:] += (plot_dy ** 2 - sectiontheta[i] ** 2) ** 0.5

        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(3, dpi=140)
        axs[0].plot((plot_y := np.linspace(0, ys[-1], len(ys))), plot_z)
        axs[0].set_aspect(1)
        axs[1].plot(plot_y, sectionL)
        axs[2].plot(plot_y, sectionM)

        axs[0].set_ylabel("z(y)")
        axs[1].set_ylabel("L(y)")
        axs[2].set_ylabel("M(y)")

        # plt.show()

        return


class CDfGudmundsson(object):

    def __init__(self, sections: DiscreteIndex, span: Hint.num,
                 altitude: Hint.num, TAS: Hint.num, geometric: bool = None,
                 atmosphere=None, N: int = None, mirror: bool = None):
        """

        Args:
            sections ():
            span ():
            altitude ():
            TAS ():
            geometric:
            atmosphere ():
            N ():
            mirror ():

        Notes:
            Section 15.3.3, Step-by-step calculation of skin friction drag
        """
        # Recast as necessary
        span = Quantity(span, "m")
        TAS = Quantity(TAS, "m s^{-1}")
        geometric = False if geometric is None else geometric
        atmosphere = ISA1975() if atmosphere is None else atmosphere
        N = 40 if N is None else int(N)
        mirror = True if mirror is None else mirror

        # Assume roughness of carefully applied matte paint
        from carpy.utility import constants as co
        kappa = co.MATERIAL.esg_roughness.mattepaint_careful.mean()

        # Linearly spaced sections in defined span (no need to mirror)
        Nsections = designate_sections(sections=sections, mirror=False, N=N)

        # Step 1) Find Viscosity of air
        mu_visc = atmosphere.mu_visc(altitude=altitude, geometric=geometric)

        # Step 2) Find Reynolds Number
        rho = atmosphere.rho(altitude=altitude, geometric=geometric)
        chords = Quantity([section.chord.x for section in Nsections], "m")
        Re = rho * TAS * chords / mu_visc

        # Step 3) Cutoff Reynolds number due to surface roughness effects
        Mach = TAS / atmosphere.c_sound(altitude=altitude, geometric=geometric)
        if Mach <= 0.7:
            Re_cutoff = 38.21 * (chords / kappa).x ** 1.053
        else:
            Re_cutoff = 44.62 * (chords / kappa).x ** 1.053 * Mach ** 1.16

        # noinspection PyArgumentList
        Re = np.vstack((Re, Re_cutoff)).min(axis=0)

        # Step 4) Compute skin friction coefficient for fully laminar/turbulent
        self._Cf_lam = 1.328 * Re ** -0.5
        self._Cf_turb = 0.455 * np.log10(Re) ** -2.58
        self._Cf_turb *= (1 + 0.144 * Mach ** 2) ** -0.65  # Compressible corr.

        # Step 5) Determine fictitious turbulent boundary layer origin point X0
        Xtr_C = 0.5  # Assume transition @X==0.5, avg of upper/lower transitions
        X0_C = 36.9 * Xtr_C ** 0.625 * Re ** -0.375

        # Step 6) Compute mixed laminar-turbulent flow skin friction coefficient
        Cf = 0.074 * Re ** -0.2 * (1 - (Xtr_C - X0_C)) ** 0.8

        # Find chords between stations Cf has been evaluated at
        b_eval = span.x / 2 if mirror is True else span.x
        ys = np.concatenate(
            ([0], moving_average(np.linspace(0, b_eval, N), w=2), [b_eval]))
        stations = np.interp(ys, [0, b_eval], [min(sections), max(sections)])
        midchords = np.array([sec.chord.x for sec in sections[stations]])
        # Find component areas of wing locally surrounding the defined sections
        Srefs = moving_average(midchords, w=2) * np.diff(ys)
        Swets = np.array([sec.aerofoil.perimeter for sec in Nsections]) * Srefs

        # Step 7) Compute skin friction drag coefficient
        Swet = Swets.sum()
        Sref = Srefs.sum()
        self._Cf = (Cf * Swets).sum() / Swet
        self._CDf = self._Cf * (Swet / Sref)

        return

    @property
    def CDf(self):
        """Wing skin friction drag coefficient, CDf."""
        return self._CDf
