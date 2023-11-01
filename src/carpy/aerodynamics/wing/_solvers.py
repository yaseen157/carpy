"""A module of various methods used to estimate wing aerodynamic performance."""
import warnings

import numpy as np

from carpy.aerodynamics.aerofoil import ThinAerofoil
from carpy.environment import LIBREF_ATM
from carpy.utility import (
    Hint, Quantity, call_count, cast2numpy, constants as co, moving_average)

__all__ = ["MixedBLDrag", "PrandtlLLT", "HorseshoeVortex"]
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


class WingSolution(object):
    """
    Template object of incompressible wing performance coefficients, which
    solvers should aim to fill for a given angle of attack.
    """
    _CL: float = NotImplemented
    _CDi: float = NotImplemented
    _CD0: float = NotImplemented
    _CD: float = NotImplemented
    _CY: float = NotImplemented
    _Cl: float = NotImplemented
    _Cm: float = NotImplemented
    _Cn: float = NotImplemented
    _Cni: float = NotImplemented
    _x_cp: float = NotImplemented

    def __init__(self, wingsections, altitude: Hint.num, TAS: Hint.num, *,
                 alpha: Hint.num = None, beta: Hint.num = None,
                 geometric: bool = None, atmosphere=None, N: int = None):
        self._wingsections = wingsections
        self._altitude = Quantity(altitude, "m")
        self._TAS = Quantity(TAS, "m s^{-1}")
        self._alpha = 0.0 if alpha is None else float(alpha)
        self._beta = 0.0 if beta is None else float(beta)
        self._geometric = False if geometric is None else geometric
        self._atmosphere = LIBREF_ATM if atmosphere is None else atmosphere
        self._Nctrlpts = 40 if N is None else int(N)
        # Track which attributes are accessed during computation:
        self._accessed = dict()
        return

    def __str__(self):
        returnstr = f"{self._wingsections} performance:\n"
        # I know it looks funny but the 7F format means these all line up fine
        returnstr += ("-" * len(returnstr)) + "\n"
        returnstr += "\n".join([
            f"| CD = {self.CD:7F} | CY  = {self.CY:7F}  | CL  = {self.CL:7F} |",
            f"|           `-->  CD0 = {self.CD0:7F}  + CDi = {self.CDi:7F} |",
            f"| Cl = {self.Cl:7F} | Cm  = {self.Cm:7F}  | Cn  = {self.Cn:7F} |",
            f"|                                   Cni = {self.Cni:7F} |"
        ]).replace("NAN", "NAN ")
        return returnstr

    def __or__(self, other):
        """Logical OR, fills missing parameters of self with other"""
        # Verify objects are of the same type, and it makes sense to logical OR
        errormsg = f"Union only applies if both objects are of {type(self)=}"
        if type(other).__bases__[0] is WingSolution:
            pass  # Both self and other are children of the WingSolution class
        elif type(other) is WingSolution:
            pass  # Self is a child of WingSolution, other *is* a WingSolution
        else:
            raise TypeError(errormsg)

        # Verify it's okay to add the predictions together:
        # ... are the solutions for common lift bodies?
        errormsg = (
            f"Can only apply union when wingsections attributes are identical "
            f"(actually got {self.wingsections, other.wingsections})"
        )
        # Use symmetric difference of wingsections sets
        memoryset_self = set([hex(id(x)) for x in self.wingsections])
        memoryset_other = set([hex(id(x)) for x in other.wingsections])
        assert len(memoryset_self ^ memoryset_other) is 0, errormsg

        # ... do the solutions have compatible flight conditions?
        # Observing the following private attributes can change their value,
        # Schrödinger's variable style, when using an interactive debugger. For
        # your own sanity, these parameters are now saved into definite vars.
        accessed_self = self._accessed
        accessed_other = other._accessed

        # If both solutions depend on a parameter, make sure they are similar
        common_attrs = set(accessed_self) & set(accessed_other)
        errormsg = (
            f"Can't do union for {type(self).__name__} objects, found that one "
            f"or more of the following values had mismatches: {common_attrs}"
        )
        for attr in common_attrs:
            if getattr(self, f"_{attr}") == getattr(other, f"_{attr}"):
                continue  # The parameters are similar, do nothing :)
            raise ValueError(errormsg)  # Uh oh, flight conditions differed!

        # Find instantiation arguments of self and other, combine them
        new_kwargs = {
            "altitude": np.nan, "TAS": np.nan,  # <-- defaults
            **{x: getattr(self, x) for x in accessed_self},
            **{x: getattr(other, x) for x in accessed_other}
        }

        # Find performance parameters of self and other, combine them
        to_combine = "CL,CDi,CD0,CD,CY,Cl,Cm,Cn,Cni,x_cp".split(",")
        result_self = {attr: getattr(self, attr) for attr in to_combine}
        result_other = {attr: getattr(other, attr) for attr in to_combine}
        result_new = {
            attr: result_self[attr]
            if ~np.isnan(result_self[attr]) else result_other[attr]
            for attr in to_combine
        }

        # Assign new performance parameters to new object
        new_soln = WingSolution(wingsections=self.wingsections, **new_kwargs)
        for (k, v) in result_new.items():
            if ~np.isnan(v):
                setattr(new_soln, f"_{k}", v)

        return new_soln

    @property
    def wingsections(self):
        """The wing sections that comprise this wing performance prediction."""
        return self._wingsections

    @property
    def altitude(self) -> Quantity:
        """Altitude of the solution."""
        self._accessed["altitude"] = True
        return self._altitude

    @property
    def TAS(self) -> Quantity:
        """True airspeed of the solution."""
        self._accessed["TAS"] = True
        return self._TAS

    @property
    def alpha(self) -> float:
        """Angle of attack of the solution."""
        self._accessed["alpha"] = True
        return self._alpha

    @property
    def beta(self) -> float:
        """Angle of sideslip of the solution."""
        self._accessed["beta"] = True
        return self._beta

    @property
    def geometric(self) -> bool:
        """Whether the altitude argument is geometric or geopotential."""
        self._accessed["geometric"] = True
        return self._geometric

    @property
    def atmosphere(self):
        """Angle of sideslip of the solution."""
        self._accessed["atmosphere"] = True
        return self._atmosphere

    @property
    def CL(self) -> float:
        """Wing coefficient of lift, CL."""
        if self._CL is NotImplemented:
            return np.nan
        return self._CL

    @property
    def CDi(self) -> float:
        """Induced component of wing's coefficient of drag, CDi."""
        if self._CDi is NotImplemented:
            return np.nan
        return self._CDi

    @property
    def CD0(self) -> float:
        """Profile component of wing's coefficient of drag, CD0."""
        if self._CD0 is NotImplemented:
            return np.nan
        return self._CD0

    @property
    def CD(self) -> float:
        """Wing coefficient of drag (profile + induced), CD."""
        if self._CD is NotImplemented:
            return self.CD0 + self.CDi
        elif np.isclose(self._CD, self.CD0 + self.CDi):
            return self._CD
        else:
            errormsg = (
                f"Total wing drag coefficient CD is not equal to the sum of "
                f"component profile and induced drags! ({self._CD} != "
                f"{self.CD0=} + {self.CDi})"
            )
            raise ValueError(errormsg)

    @property
    def CY(self) -> float:
        """Wing coefficient of side/lateral force (+ve := slip right), CY."""
        if self._CY is NotImplemented:
            return np.nan
        return self._CY

    @property
    def Cl(self) -> float:
        """Wing rolling moment coefficient (+ve := roll right), Cl."""
        if self._Cl is NotImplemented:
            return np.nan
        return self._Cl

    @property
    def Cm(self) -> float:
        """Wing pitching moment coefficient (+ve := pitch up), Cm."""
        if self._Cm is NotImplemented:
            return np.nan
        return self._Cm

    @property
    def Cn(self) -> float:
        """Wing yawing moment coefficient (+ve := right rudder), Cn."""
        if self._Cn is NotImplemented:
            return np.nan
        return self._Cn

    @property
    def Cni(self) -> float:
        """Wing induced yawing moment coefficient (+ve := right rudder), Cni."""
        if self._Cni is NotImplemented:
            return np.nan
        return self._Cni

    @property
    def x_cp(self) -> float:
        """
        Chordwise location of centre of pressure, as a fraction of the root
        chord behind the leading edge.
        """
        if self._x_cp is NotImplemented:
            return np.nan
        return self._x_cp


# ============================================================================ #
# Public (solution) classes
# ---------------------------------------------------------------------------- #


class MixedBLDrag(WingSolution):
    """
    Method for predicting viscous drag due to skin friction effects.

    References:
        Gudmundsson, S., "General Aviation Aircraft Design: Applied Methods and
            Procedures", Butterworth-Heinemann, 2014, pp.675-685.
    """

    def __init__(self, wingsections, altitude: Hint.num, TAS: Hint.num,
                 Ks: Hint.num = None, **kwargs):
        # Super class call
        super().__init__(wingsections, altitude, TAS, **kwargs)

        # Recast as necessary
        TAS = Quantity(self.TAS, "m s^{-1}")
        Ks = co.MATERIAL.roughness_Ks.paint_matte_smooth if Ks is None else Ks
        Ks = np.mean(Ks)
        self._CDf_CD0 = 0.85  # Assume 85% of profile drag is friction

        # Bail early if necessary (why would anyone evaluate zero speed drag?)
        if TAS == 0:
            self._Cf_laminar = np.nan
            self._Cf_turbulent = np.nan
            self._Cf = np.nan
            self._CDf = np.nan
            self._CD0 = self._CDf / self.CDf_CD0
            return

        # Linearly spaced sections defined in the span
        sections = wingsections.spansections(N=(N := self._Nctrlpts))

        # Step 1) Find viscosity of air
        fltconditions = dict([
            ("altitude", self.altitude), ("geometric", self.geometric)])
        mu_visc = self.atmosphere.mu_visc(**fltconditions)

        # Step 2) Compute Reynolds number
        rho = self.atmosphere.rho(**fltconditions)
        chords = Quantity([sec.chord.x for sec in sections], "m")
        Re = rho * TAS * chords / mu_visc

        # Step 3) Cutoff Reynolds number due to surface roughness effects
        Mach = TAS / self.atmosphere.c_sound(**fltconditions)
        if Mach <= 0.7:
            Re_cutoff = 38.21 * (chords / Ks).x ** 1.053
        else:
            Re_cutoff = 44.62 * (chords / Ks).x ** 1.053 * Mach ** 1.16
        Re = np.vstack((Re, Re_cutoff)).min(axis=0)

        # Step 4) Compute skin friction coefficient for fully laminar/turbulent
        self._sectionCf_laminar = 1.328 * Re ** -0.5
        self._sectionCf_turbulent = 0.455 * np.log10(Re) ** -2.58
        # Compressiblity correction
        self._sectionCf_turbulent *= (1 + 0.144 * Mach ** 2) ** -0.65

        # Step 5) Determine fictitious turbulent boundary layer origin point X0
        Xtr_C = 0.5  # Assume transition @X==0.5, avg of upper/lower transitions
        X0_C = 36.9 * Xtr_C ** 0.625 * Re ** -0.375

        # Step 6) Compute mixed laminar-turbulent flow skin friction coefficient
        # Young's method:
        Cfs = 0.074 * Re ** -0.2 * (1 - (Xtr_C - X0_C)) ** 0.8

        # Find the chords between the stations at which Cf is evaluated
        mid_chords = moving_average([sec.chord.x for sec in sections])
        # Discretise the Sref of the wing into components centred on mid_chords
        mid_Srefs = mid_chords * wingsections.b.x / (N - 1)
        mid_perim = moving_average([sec.aerofoil.perimeter for sec in sections])
        mid_Swets = mid_Srefs * mid_perim

        # Step 7) Compute skin friction drag coefficient
        Swet = mid_Swets.sum()
        Sref = mid_Srefs.sum()
        self._Cf = (moving_average(Cfs) * mid_Swets).sum() / Swet
        self._CDf = self._Cf * (Swet / Sref)

        # Assignments
        self._CD0 = self._CDf / self.CDf_CD0

        return

    # @property
    # def Cf_laminar(self) -> float:
    #     """100% Laminar limit of skin friction coefficient, Cflaminar."""
    #     return self._Cf_laminar
    #
    # @property
    # def Cf_turbulent(self) -> float:
    #     """100% Turbulent limit of skin friction coefficient, Cfturbulent."""
    #     return self._Cf_turbulent

    @property
    def Cf(self) -> float:
        """Skin friction coefficient, Cf."""
        return self._Cf

    @property
    def CDf(self) -> float:
        """Coefficient of friction drag, CDf."""
        return self._CDf

    @property
    def CDf_CD0(self) -> float:
        """Proportion of profile drag that is composed of skin friction drag."""
        return self._CDf_CD0

    @CDf_CD0.setter
    def CDf_CD0(self, value):
        self._CDf_CD0 = float(value)
        self._CD0 = self.CDf / self.CDf_CD0  # Update profile drag estimate
        return


class PrandtlLLT(WingSolution):
    """
    Prandtl's lifting line (a.k.a. Lanchester-Prandtl wing) theory.

    For use in incompressible, inviscid, steady flow regimes with moderate to
    high aspect ratio, unswept wings.
    """

    def __init__(self, wingsections, altitude: Hint.num, TAS: Hint.num,
                 **kwargs):
        # Super class call
        super().__init__(wingsections, altitude, TAS, **kwargs)

        # Library limitations
        self._CY = 0  # This is an assumption, I'm not actually sure...
        if (beta := self.beta) != 0:
            raise ValueError(f"Sorry, non-zero beta is unsupported ({beta=})")

        # Cosine distribution of wing sections (skip first and last section)
        theta0 = np.linspace(0, np.pi, (N := self._Nctrlpts) + 2)[1:-1]
        sections = wingsections.spansections(N=N + 2, dist_type="cosine")[1:-1]

        # Station: geometric parameters
        chord = [sec.chord.x for sec in sections]
        # Station: aerodynamic parameters
        alpha_zl, Clalpha = [], []
        for i, sec in enumerate(sections):
            aoa = self.alpha + sec.twist  # Effective angle of attack
            soln_thinaero = ThinAerofoil(aerofoil=sec.aerofoil, alpha=aoa)
            alpha_zl.append(soln_thinaero.alpha_zl - sec.twist)  # Eff. zerolift
            Clalpha.append(soln_thinaero.Clalpha)

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
        self._CL = np.pi * wingsections.AR * matX[0]
        self._CDi = self.CL ** 2 / (np.pi * e * wingsections.AR)

        return


class HorseshoeVortex(WingSolution):
    """
    Horseshoe Vortex Method, with N vortices.

    For use in incompressible, inviscid, steady flow regimes.

    Notes:
        Currently assumes all lifting sections have a lift-slope of 2 pi.

    """

    def __init__(self, wingsections, altitude: Hint.num, TAS: Hint.num,
                 **kwargs):
        super().__init__(wingsections, altitude, TAS, **kwargs)

        # Problem setup
        # ... create rotation matrix for angle of attack and of sideslip
        cos_alpha, sin_alpha = np.cos(-(alpha := self.alpha)), np.sin(-alpha)
        rot_alpha = np.array([
            [cos_alpha, 0, sin_alpha],
            [0, 1, 0],  # About the Y-axis
            [-sin_alpha, 0, cos_alpha]
        ])
        cos_beta, sin_beta = np.cos(self.beta), np.sin(self.beta)
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

            solution = ThinAerofoil(
                aerofoil=sections[i].aerofoil,
                alpha=alpha + sections[i].twist  # Effective AoA
            )
            alpha_zl = solution.alpha_zl - sections[i].twist  # Effective ZL

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
        # self._CY = self._sectionCY.sum()  # Unsure if direction is correct...
        self._CL = self._sectionCL.sum()

        return

# class Cantilever1DStatic(WingSolution):
#
#     def __init__(self, sections, spar, span: Hint.num, alpha: Hint.num,
#                  lift: Hint.num, N: int = None, mirror: bool = None,
#                  model: str = None):
#         # Define compatability parameters
#         supported_models = [HorseshoeVortex]
#         supported_models = dict(zip(
#             [x.__name__ for x in supported_models],
#             supported_models
#         ))
#
#         # Recast as necessary
#         kwargs = {
#             "sections": sections, "spar": spar, "span": span, "alpha": alpha,
#             "lift": lift, "N": 40 if N is None else int(N),
#             "mirror": True if mirror is None else mirror,
#             "model": model
#         }
#         if kwargs["model"] is None:  # Model unspecified
#             kwargs["model"] = HorseshoeVortex
#         elif supported_models.get(model) is None:  # Model unsupported
#             errormsg = f"{model=} is unsupported, try one of {supported_models}"
#             raise ValueError(errormsg)
#
#         # Evaluate aerodynamic model
#         # noinspection PyUnresolvedReferences
#         soln_aero = kwargs["model"](
#             **{  # Iterate over kwargs, and pass only those which appear in init
#                 k: v for (k, v) in kwargs.items()
#                 if k in kwargs["model"].__init__.__annotations__  # <-Typehints!
#             }
#         )
#
#         # Find out if solution is mirrored
#         ys = soln_aero.xyz_cp[:, 1]
#         if kwargs["mirror"] is True:
#             root_i = int(np.ceil(len(ys) / 2))
#             ys = ys[root_i:]
#         else:
#             root_i = 0
#
#         # Lift and moment component distribution
#         sectionL = lift * (soln_aero.Cl[root_i:] / soln_aero.CL)
#         sectionM = sectionL * ys
#
#         # Flexural modulus of carbon fibre (?)
#         E_rect = 60e9  # e9 == GPa
#         # Subscript x as it's the effect of material distributed about x-axis
#         t_wall = 1e-3
#         b_spar = 50e-3
#         h0_spar = 100e-3
#         h1_spar = 20e-3
#
#         def Ix_rect(y):
#             """Variable cross-section, hollow rectangular spar."""
#             y = np.abs(y)
#             h_spar = np.interp(y, [0, 10], [h0_spar, h1_spar])
#             Ixx_o = b_spar * h_spar ** 3 / 12
#             Ixx_i = (b_spar - 2 * t_wall) * (h_spar - 2 * t_wall) ** 3 / 12
#             return Ixx_o - Ixx_i
#
#         EI = E_rect * Ix_rect(ys)
#
#         # Distribution of angular increments with each section
#         sectiontheta = sectionM / EI
#
#         # Distibution of vertical displacement with each section
#         sectionnu = np.cumsum(sectiontheta)
#
#         plot_dy, = np.diff(ys[0:2])
#         plot_y = np.zeros_like(sectionnu)
#         plot_z = np.copy(sectionnu)
#
#         for i in range(1, len(sectiontheta)):
#             plot_y[i:] += (plot_dy ** 2 - sectiontheta[i] ** 2) ** 0.5
#
#         from matplotlib import pyplot as plt
#
#         fig, axs = plt.subplots(3, dpi=140)
#         axs[0].plot((plot_y := np.linspace(0, ys[-1], len(ys))), plot_z)
#         axs[0].set_aspect(1)
#         axs[1].plot(plot_y, sectionL)
#         axs[2].plot(plot_y, sectionM)
#
#         axs[0].set_ylabel("z(y)")
#         axs[1].set_ylabel("L(y)")
#         axs[2].set_ylabel("M(y)")
#
#         # plt.show()
#
#         return
