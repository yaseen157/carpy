"""Methods for generating and optimising wing planforms."""
from functools import partial, cached_property

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import griddata

from carpy.aerodynamics.aerofoil import Aerofoil
from carpy.structures import DiscreteIndex
from carpy.utility import (
    CacheClr, Hint, Quantity, cast2numpy, collapse_array, isNone)

__all__ = ["WingSection", "WingSections", "WingPlane", "NewWingPlane"]
__author__ = "Yaseen Reza"


class WingSection(object):
    """
    Class for modelling wing cross-sections, as aligned with wing buttock lines.

    A wing section is a 2D cross-sectional slice of the 3D wing structure, and
    inclined at the local angle of dihedral. Sections are defined through a
    non-dimensional aerofoil geometry, the chord length, and the applied twist,
    sweep, and dihedral angles.
    """

    def __init__(self, aerofoil: Aerofoil = None, chord: Hint.num = None,
                 twist: Hint.num = None):
        self._aerofoil = aerofoil
        self._twist = 0.0 if twist is None else float(twist)
        self.chord = 1.0 if chord is None else float(chord)
        self._sweep = None
        self._dihedral = None
        return

    def __add__(self, other):
        # Typechecking
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot add {type(self)=} to {type(other)=}")

        new_object = type(self)(
            aerofoil=self.aerofoil + other.aerofoil,
            chord=self.chord + other.chord,
            twist=self.twist + other.twist
        )
        return new_object

    def __mul__(self, other):
        # Typechecking
        if not isinstance(other, Hint.num.__args__):
            raise TypeError(f"Cannot multiply {type(self)=} by {type(other)=}")
        # New non-dimensional profile, chord length, and angle of twist
        new_aerofoil = other * self.aerofoil
        new_chord = other * self.chord
        new_theta = other * self.twist

        new_object = type(self)(
            aerofoil=new_aerofoil,
            chord=new_chord,
            twist=new_theta
        )
        return new_object

    def deepcopy(self):
        """Returns a deep copy of self (no values are shared in memory)."""
        return type(self)(aerofoil=self._aerofoil, twist=self._twist)

    @property
    def aerofoil(self):
        """Non-dimensional aerofoil object, attached to the station."""
        return self._aerofoil

    @property
    def chord(self) -> Quantity:
        """
        The chord length of the station's aerofoil.

        Returns:
            Reference chord length of the aerofoil at this wing station.

        """
        return self._chord

    @chord.setter
    def chord(self, value):
        self._chord = Quantity(value, "m")
        return

    @property
    def twist(self) -> float:
        """
        The station's geometric angle of twist, relative to wing root. For a
        horizontal wing, a positive value indicates wash-in (station has an
        angle of incidence greater than the wing root), and a negative value
        indicates wash-out.

        Returns:
            Geometric angle of twist of the wing station.

        """
        return self._twist

    @twist.setter
    def twist(self, value):
        self._twist = float(value)
        return

    @property
    def sweep(self) -> float:
        """
        The sweep angle of the leading edge. A positive value indicates sweep in
        the traditional sense (outboard leading edge is behind inboard leading
        edge), and a negative value indicates forward- or reverse-sweep.

        Returns:
            Geometric angle of sweep of the wing station's leading edge.

        """
        return self._sweep

    @sweep.setter
    def sweep(self, value):
        self._sweep = float(value)
        return

    @property
    def dihedral(self) -> float:
        """
        The angle between the station plane's normal vector and the vehicle
        plane formed of longitudinal (X) and lateral (Y) axes.

        Returns:
            Geometric angle of dihedral of the wing station.

        """
        return self._dihedral

    @dihedral.setter
    def dihedral(self, value):
        self._dihedral = float(value)
        return


class WingSections(DiscreteIndex, CacheClr):

    def __init__(self, b: Hint.num, mirrored: bool = None):
        """
        Args:
            b: The span of the wing.
            mirrored: Whether the wing sections should be mirrored around the
                smallest numbered station. Optional, defaults to True (the
                wing is mirrored around a centreline).
        """
        # Super class call
        super().__init__()

        # Recast as necessary
        mirrored = True if mirrored is None else mirrored

        # Library limitations
        if mirrored is False:
            errormsg = f"Sorry, this feature is not ready: {mirrored=}"
            raise NotImplementedError(errormsg)

        # Assignments
        self._b = Quantity(b, "m")
        self._mirrored = mirrored

        return

    def __getitem__(self, key):
        nd_sections = super().__getitem__(key)

        if isinstance(key, slice):
            for slice_bound in [key.start, key.stop]:
                if slice_bound not in self and not isNone(slice_bound):
                    errormsg = (
                        f"Slice is bounded by key that has no associated value."
                        f" Try creating a new element of {self} with the index "
                        f"{slice_bound}"
                    )
                    raise RuntimeError(errormsg)

        # Cast to list temporarily, if necessary
        if not isinstance(nd_sections, Hint.iter.__args__):
            nd_sections = [nd_sections]

        # Missing sweep and dihedral angles should be inherited from inboard
        for i, aerofoil in enumerate(nd_sections):
            # If aerofoil is derived, it has parents (one of which is inboard)
            if hasattr(aerofoil, "_parents"):
                parent = self[min(getattr(aerofoil, "_parents"))]
                # Assign private vars, skips float typecasting for 'None' values
                aerofoil._sweep = parent.sweep
                aerofoil._dihedral = parent.dihedral
                delattr(aerofoil, "_parents")

        # If sliced, return an array
        if isinstance(key, slice):
            return nd_sections
        return collapse_array(nd_sections)

    def spansections(self, N: int = None, dist_type: str = None) -> tuple:
        """
        Return interpolated geometrical sections along the wing's full span.
        Args:
            N: Number of sections to the wing to return. Optional, defaults to
                11 (sections).
            dist_type: Distribution type of the sections. Available options are
                "linear" for linear spacing, or "cosine" for cosine spacing.
                Optional, defaults to linear spacing.

        Returns:
            A tuple of N WingSection objects.

        """
        # Recast as necessary
        N = 11 if N is None else int(N)
        if dist_type is None:
            dist_type = "linear"
        elif dist_type not in ["linear", "cosine"]:
            pass

        # If linear spacing is requested
        if dist_type == "linear":

            # If the wing sections require mirroring
            if self._mirrored is True:
                if N % 2:  # If the number of sections is odd
                    # Centreline, to starboard tip
                    sections2mirror = self[::N // 2 + 1]
                    sections2return = sections2mirror[:0:-1] + sections2mirror
                else:  # The number of sections is even
                    # Just beyond centreline, to staboard tip
                    sections2mirror = self[::N][1::2]
                    sections2return = sections2mirror[::-1] + sections2mirror
            else:
                sections2return = self[::N]  # Asymmetric wing is trivial

        # If cosine spacing is requested
        elif dist_type == "cosine":

            # If the wing sections require mirroring
            if self._mirrored is True:
                theta0 = np.linspace(0, np.pi, N)
                ys = -np.cos(theta0) * (semispan := self.b.x / 2)
                section_ids = \
                    np.interp(np.abs(ys), [0, semispan], [min(self), max(self)])
                sections2return = self[section_ids]

            else:
                errormsg = f"Asymmetric wings with {dist_type=} are unsupported"
                raise NotImplementedError(errormsg)

        else:
            raise NotImplementedError(f"Sorry, {dist_type=} is unrecognised.")

        return tuple(sections2return)

    @property
    def b(self) -> Quantity:
        """Full span of the wing, b."""
        return self._b

    @property
    def mirrored(self) -> bool:
        """Whether the wing is symetrically mirrored around the root station."""
        return self._mirrored

    @property
    def AR(self) -> float:
        """Aspect ratio of the wing, AR."""
        AR = float(self.b / self.MGC)
        return AR

    @cached_property
    def MAC(self) -> Quantity:
        """Mean aerodynamic chord, MAC."""
        sections = self.spansections(N=(N := 1001), dist_type="linear")
        chords = np.array([sec.chord.x for sec in sections])

        MAC = Quantity(
            simpson(chords ** 2, dx=self.b.x / (N - 1)) / self.Sref.x, "m")

        return MAC

    @cached_property
    def MGC(self) -> Quantity:
        """Mean geometric chord, MGC."""
        sections = self.spansections(N=1001, dist_type="linear")
        MGC = Quantity(np.mean([section.chord.x for section in sections]), "m")
        return MGC

    @property
    def SMC(self) -> Quantity:
        """Standard mean chord, SMC."""
        return self.MGC

    @property
    def Sref(self) -> Quantity:
        """Reference area of the wing, Sref."""
        Sref = self.b * self.MGC
        return Sref

    @cached_property
    def Swet(self) -> Quantity:
        """Wetted area of the wing, Swet."""
        sections = self.spansections(N=(N := 1001), dist_type="linear")
        perimeters = np.array([sec.aerofoil.perimeter for sec in sections])
        chords = np.array([sec.chord.x for sec in sections])
        nd_wetlengths = perimeters * chords

        Swet = Quantity(
            simpson(nd_wetlengths, dx=self.b.x / (N - 1)), "m^{2}")

        return Swet


# ============================================================================ #
# Public-facing WingPlane class
# ---------------------------------------------------------------------------- #
class WingPlane(object):
    """
    A class for modelling wingplanes and low-speed aerodynamic performance.
    """
    _f_CL = NotImplemented
    _f_CDi = NotImplemented
    _f_CD0 = NotImplemented
    _f_CD = NotImplemented
    _f_CY = NotImplemented
    _f_Cl = NotImplemented
    _f_Cm = NotImplemented
    _f_Cn = NotImplemented
    _f_Cni = NotImplemented
    _f_x_cp = NotImplemented

    def __init__(self, wingsections: WingSections, alpha: Hint.nums, *,
                 beta: Hint.nums = None, CL: Hint.nums = None,
                 CDi: Hint.nums = None, CD0: Hint.nums = None,
                 CD: Hint.nums = None, CY: Hint.nums = None,
                 Cl: Hint.nums = None, Cm: Hint.nums = None,
                 Cn: Hint.nums = None, Cni: Hint.nums = None,
                 x_cp: Hint.nums = None):
        """
        Args:
            wingsections: A sections object describing the wing geometry.
            alpha: Angle of attack. of the wing.

        Keyword Args:
            beta: Angle of sideslip of the wing. Optional, defaults to 0.
            CL: Wing's coefficient of lift. Optional.
            CDi: Induced component of wing's coefficient of drag. Optional.
            CD0: Profile component of wing's coefficient of drag. Optional.
            CD: Wing's coefficient of drag (profile + induced). Optional.
            CY: Wing's coefficient of side/lateral force (+ve := slip right).
                Optional.
            Cl: Wing's rolling moment coefficient (+ve := roll right). Optional.
            Cm: Wing's pitching moment coefficient (+ve := pitch up).
                 Optional.
            Cn: Wing's yawing moment coefficient (+ve := right rudder).
                Optional.
            Cni: Wing's induced yawing moment coefficient. Optional.
            x_cp: Location of centre of pressure, coincident with the chordline
                of the wing's root station, as a fraction of said station's
                length. Optional.

        Notes:
            -   Force coefficients are non dimensionalised with the wing's
                planform area.
            -   Moment coefficients are non dimensionalised with the wing's
                mean aerodynamic chord.

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)
        kwds = dict([
            ("CL", CL), ("CDi", CDi), ("CD0", CD0), ("CD", CD), ("CY", CY),
            ("Cl", Cl), ("Cm", Cm), ("Cn", Cn), ("Cni", Cni), ("x_cp", x_cp)
        ])
        del CL, CDi, CD0, CD, CY, Cl, Cm, Cn, Cni, x_cp  # Prevent accidents
        for (k, v) in kwds.items():
            kwds[k] = v if isNone(v) else cast2numpy(v)

        # Library limitations
        if not isNone(beta) and np.unique(beta) != 0.0:
            # beta = cast2numpy(0.0 if isNone(beta) else beta)
            raise ValueError("Sorry, non-zero beta angles are unsupported.")

        # Sanity check on the drag components: All components are presented
        if not any(isNone(kwds["CDi"], kwds["CD0"], kwds["CD"])):
            dragconsistent = np.isclose(
                kwds["CD"], kwds["CDi"] + kwds["CD0"], atol=1e-5)
            assert dragconsistent.all()
        # CDi is missing but the others are present
        elif isNone(kwds["CDi"]) and not any(isNone(kwds["CD0"], kwds["CD"])):
            kwds["CDi"] = kwds["CD"] - kwds["CD0"]
        # CD0 is missing but the others are present
        elif isNone(kwds["CD0"]) and not any(isNone(kwds["CDi"], kwds["CD"])):
            kwds["CD0"] = kwds["CD"] - kwds["CDi"]
        # CD is missing but the others are present
        elif isNone(kwds["CD"]) and not any(isNone(kwds["CDi"], kwds["CD0"])):
            kwds["CD"] = kwds["CDi"] + kwds["CD0"]

        # Assign geometry
        self._wingsections = wingsections

        # Autopopulate functions
        for (k, v) in kwds.items():
            if v is None:
                continue  # Skip any keyword arguments that weren't assigned
            # Otherwise, create a private interpolation function
            setattr(self, f"_f_{k}", partial(griddata, points=alpha, values=v))

        return

    @property
    def wingsections(self) -> WingSections:
        """The 3D geometry of the wing."""
        return self._wingsections

    def CL(self, alpha: Hint.nums, **kwargs) -> np.ndarray:
        """
        Wing's interpolated, incompressible coefficient of lift.

        Args:
            alpha: Angle of attack.
            **kwargs: Extra kwargs to pass to scipy.interpolate.griddata.

        Returns:
            Coefficient of lift.

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)
        CL = self._f_CL(xi=alpha, **kwargs)
        return CL

    def CDi(self, alpha: Hint.nums, **kwargs) -> np.ndarray:
        """
        Wing's interpolated, incompressible coefficient of drag (induced
        component).

        Args:
            alpha: Angle of attack.
            **kwargs: Extra kwargs to pass to scipy.interpolate.griddata.

        Returns:
            Coefficient of drag (induced component).

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)
        CDi = self._f_CDi(xi=alpha, **kwargs)
        return CDi

    def CD0(self, alpha: Hint.nums, **kwargs) -> np.ndarray:
        """
        Wing's interpolated, incompressible coefficient of drag (profile
        component).

        Args:
            alpha: Angle of attack.
            **kwargs: Extra kwargs to pass to scipy.interpolate.griddata.

        Returns:
            Coefficient of drag (profile component).

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)
        CD0 = self._f_CD0(xi=alpha, **kwargs)
        return CD0

    def CD(self, alpha: Hint.nums, **kwargs) -> np.ndarray:
        """
        Wing's interpolated, incompressible coefficient of drag (profile +
        induced).

        Args:
            alpha: Angle of attack.
            **kwargs: Extra kwargs to pass to scipy.interpolate.griddata.

        Returns:
            Coefficient of drag (profile + induced).

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)
        CD = self._f_CD(xi=alpha, **kwargs)
        return CD

    def CY(self, alpha: Hint.nums, **kwargs) -> np.ndarray:
        """
        Wing's interpolated, incompressible coefficient of side/lateral force.

        Args:
            alpha: Angle of attack.
            **kwargs: Extra kwargs to pass to scipy.interpolate.griddata.

        Returns:
            Coefficient of side/lateral force.

        Notes:
            +ve indicates the wing is slipping to the right.

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)
        CY = self._f_CY(xi=alpha, **kwargs)
        return CY

    def Cl(self, alpha: Hint.nums, **kwargs) -> np.ndarray:
        """
        Wing's interpolated, incompressible rolling moment coefficient.

        Args:
            alpha: Angle of attack.
            **kwargs: Extra kwargs to pass to scipy.interpolate.griddata.

        Returns:
            Rolling moment coefficient.
        
        Notes:
            +ve indicates rolling to the right.

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)
        Cl = self._f_Cl(xi=alpha, **kwargs)
        return Cl

    def Cm(self, alpha: Hint.nums, **kwargs) -> np.ndarray:
        """
        Wing's interpolated, incompressible pitching moment coefficient.

        Args:
            alpha: Angle of attack.
            **kwargs: Extra kwargs to pass to scipy.interpolate.griddata.

        Returns:
            Pitching moment coefficient.

        Notes:
            +ve indicates pitching up (nose up, tail down).

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)
        Cm = self._f_Cm(xi=alpha, **kwargs)
        return Cm

    def Cn(self, alpha: Hint.nums, **kwargs) -> np.ndarray:
        """
        Wing's interpolated, incompressible yawing moment coefficient.

        Args:
            alpha: Angle of attack.
            **kwargs: Extra kwargs to pass to scipy.interpolate.griddata.

        Returns:
            Yawing moment coefficient.

        Notes:
            +ve indicates steering to the right (as in right rudder).

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)
        Cn = self._f_Cn(xi=alpha, **kwargs)
        return Cn

    def Cni(self, alpha: Hint.nums, **kwargs) -> np.ndarray:
        """
        Wing's interpolated, incompressible induced yawing moment coefficient.

        Args:
            alpha: Angle of attack.
            **kwargs: Extra kwargs to pass to scipy.interpolate.griddata.

        Returns:
            Induced yawing moment coefficient.

        Notes:
            +ve indicates steering to the right (as in right rudder).

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)
        Cni = self._f_Cni(xi=alpha, **kwargs)
        return Cni

    def x_cp(self, alpha: Hint.nums, **kwargs) -> np.ndarray:
        """
        Wing's interpolated, incompressible centre of pressure location in the
        x-direction. The location is given along the root chord of the wing, as
        a fraction of the length of the root chord.

        Args:
            alpha: Angle of attack.
            **kwargs: Extra kwargs to pass to scipy.interpolate.griddata.

        Returns:
            Centre of pressure location.

        Notes:
            +ve indicates steering to the right (as in right rudder).

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)
        x_cp = self._f_x_cp(xi=alpha, **kwargs)
        return x_cp


class NewWingPlane(object):
    """
    A class of methods for attributing aerodynamic performance to wing planes.
    """

    @classmethod
    def from_XFLR5file(cls, wingsections: WingSections, filepath):
        """
        Return a wingplane object, given geometry and XFLR5 performance data.

        Args:
            wingsections: A sections object describing the wing geometry.
            filepath: The path to a file containing XFLR5 wing polar data.

        Returns:
            A WingPlane object.

        """
        # The first 5 lines are descriptive, and not data we want right now
        df = pd.read_csv(filepath, sep=r"\s+", skiprows=5, engine="python")

        try:
            betas, QInfs = df[["Beta", "QInf"]].to_numpy().T
            # Verify there is only 1 element by unpacking
            beta, = np.unique(betas)
            _, = np.unique(QInfs)
            # Verify that sideslip angle is zero.
            assert beta == 0
        except KeyError:
            errormsg = (
                f"Data could be read, but this could be incomplete XFLR5 polar "
                f"data. Was expecting to find keys 'Beta' and 'QInf'."
            )
            raise ValueError(errormsg)
        except ValueError:
            errormsg = (
                f"Found XFLR5 data, but the case is unsupported. Please ensure "
                f"that freestream sideslip angle and velocity are constant."
            )
            raise ValueError(errormsg)
        except AssertionError:
            errormsg = (
                f"Found XFLR5 data, but the case is unsupported. Please ensure "
                f"that freestream sideslip angle is zero (beta=0)."
            )
            raise ValueError(errormsg)

        mywingplane = WingPlane(
            wingsections=wingsections, alpha=np.radians(df["alpha"]),
            beta=np.radians(df["Beta"]), CL=df["CL"], CDi=df["CDi"],
            CD0=df["CDv"], CD=df["CD"], CY=df["CY"], Cl=df["Cl"], Cm=df["Cm"],
            Cn=df["Cn"], Cni=df["Cni"], x_cp=df["XCP"]
        )

        return mywingplane

# class WingPlane(object):
#     """
#     Class for modelling wing planes.
#     """
#
#     def __init__(self, span: Hint.num, mirror: bool = None):
#         """
#         Args:
#             span: The full span of the wing, as determined from planform view.
#             mirror: Whether or not to mirror the wing's stations about the
#                 centreline.
#         """
#         self._b = span
#         self._mirror = True if mirror is None else mirror
#         self._spar = None
#
#         return
#
#     @property
#     def parametric_spar(self) -> Hint.func:
#         """
#         A parameterised definition of the wingspar's (composite) geometry and
#         material selection(s). Accepts arguments for the maximal 'height' and
#         'width' of a spar section.
#
#         Returns:
#             Spar section object, with geometric properties pre-computed.
#
#         """
#         if self._spar is None:
#             raise NotImplementedError("Spar definitions has not yet been given")
#         return self._spar
#
#     @parametric_spar.setter
#     def parametric_spar(self, value):
#         if not callable(value):
#             errormsg = (
#                 f"section_spar.setter is expecting to be given 'function(y)', "
#                 f"actually got section_spar = {value} (invalid {type(value)=})"
#             )
#             raise TypeError(errormsg)
#         self._spar = value
#         return
#
#     @parametric_spar.deleter
#     def parametric_spar(self):
#         self._spar = None
#         return

# if __name__ == "__main__":
#     from sectionproperties.pre.library import steel_sections, primitive_sections
#     from sectionproperties.pre.pre import Material
#     from sectionproperties.analysis.section import Section
#
#     steel = Material(
#         name='Steel', elastic_modulus=200e9, poissons_ratio=0.3,
#         density=7.85e3, yield_strength=500e6, color='grey'
#     )
#     timber = Material(
#         name='Timber', elastic_modulus=8e9, poissons_ratio=0.35,
#         density=6.5e2, yield_strength=20e6, color='burlywood'
#     )
#
#
#     def sections(height: float, width: float):
#         """
#
#         Args:
#             height: Maximum allowable height of the spar.
#             width: Maximum allowable width of the spar.
#
#         Returns:
#
#         """
#         # Compute parameterised dimensions
#         core_y = height - (2 * width)
#         tube_od = width
#         tube_wt = 1e-3
#
#         # STEP 1: Create component section geometries (and attach materials)
#         rod_kwargs = {"d": tube_od, "t": tube_wt, "n": 100, "material": steel}
#         core_kwargs = {"b": rod_kwargs["d"], "d": core_y, "material": timber}
#         rod = steel_sections.circular_hollow_section(**rod_kwargs)
#         core = primitive_sections.rectangular_section(**core_kwargs)
#
#         # STEP 2: Create a compound geometry, and convert into a section object
#         section_geometry = (
#                 rod.shift_section(0, (core_y + rod_kwargs["d"]) / 2)
#                 + core.shift_section(-rod_kwargs["d"] / 2, -core_y / 2)
#                 + rod.shift_section(0, -(core_y + rod_kwargs["d"]) / 2)
#         )
#         section_geometry.create_mesh(mesh_sizes=[1e-3])
#         section = Section(section_geometry)
#
#         # STEP 3: Calculate the geometric properties of the section
#         section.calculate_geometric_properties()
#
#         return section
#
#
#     section = sections(height=0.1, width=12e-3)
#     section.plot_mesh()
