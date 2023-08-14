"""Methods for generating and optimising wing planforms."""
import numpy as np

from carpy.aerodynamics.aerofoil import NDAerofoil
from carpy.structures import DiscreteIndex
from carpy.utility import Hint, cast2numpy, collapse1d, isNone

__all__ = ["NDWingStation", "WingStations"]
__author__ = "Yaseen Reza"


class NDWingStation(object):
    """
    Class for modelling non-dimensional wing cross-sections (wing stations).

    A wing station is a 2D cross-sectional slice of the 3D wing structure, in a
    plane perpendicular to that of the leading edge. As a result, the geometry
    of the section is independent of any applied sweep or dihedral - despite
    the station itself having sweep/dihedral.
    """

    def __init__(self, aerofoil: NDAerofoil = None, twist: Hint.num = None):
        self._aerofoil = aerofoil
        self._twist = 0.0 if twist is None else float(twist)
        self._sweep = None
        self._dihedral = None
        return

    def __add__(self, other):
        # Typechecking
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot add {type(self)=} to {type(other)=}")

        new_object = type(self)(
            aerofoil=self.aerofoil + other.aerofoil,
            twist=self.twist + other.twist
        )
        return new_object

    def __mul__(self, other):
        # Typechecking
        if not isinstance(other, Hint.num.__args__):
            raise TypeError(f"Cannot multiply {type(self)=} by {type(other)=}")
        # New non-dimensional profile and angle of twist
        new_aerofoil = other * self.aerofoil
        new_theta = other * self.twist

        new_object = type(self)(
            aerofoil=new_aerofoil,
            twist=new_theta
        )
        return new_object

    @property
    def aerofoil(self):
        return self._aerofoil

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
        The angle between the plane of the wing station and the vehicle's
        longitudinal axis. A positive value indicates sweep in the traditional
        sense (outboard leading edge is behind inboard leading edge), and a
        negative value indicates forward- or reverse-sweep.

        Returns:
            Geometric angle of sweep of the wing station.

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

    @property
    def alpha_zl(self) -> float:
        """Station's angle of attack for zero-lift, relative to wing root."""
        alpha_zl = self.aerofoil.alpha_zl - self.twist
        return alpha_zl

    @alpha_zl.setter
    def alpha_zl(self, value):
        errormsg = (
            f"'alpha_zl' of a station is not configurable as it is a derived "
            f"property of the station's aerofoil zero lift angle and twist. "
            f"Try modifying the station's aerofoil or twist parameters instead"
        )
        raise AttributeError(errormsg)

    def Clalpha(self, alpha: Hint.nums) -> np.ndarray:
        """
        The station's lift-curve slope.

        Args:
            alpha: Angle of attack at the wing root.

        Returns:
            Station's lift-curve slope.

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)

        local_alpha = alpha + self.twist
        Clalpha = self.aerofoil.Clalpha(alpha=local_alpha)
        return Clalpha

    def Cl(self, alpha: Hint.nums) -> np.ndarray:
        """
        The station's lift coefficient.

        Args:
            alpha: Angle of attack at the wing root.

        Returns:
            Station's lift coefficent.

        """
        # Recast as necessary
        alpha = cast2numpy(alpha)

        local_alpha = alpha + self.twist
        Cl = self.aerofoil.Cl(alpha=local_alpha)
        return Cl


class WingStations(DiscreteIndex):

    def __init__(self, span: Hint.num):
        """
        Args:
            span: The full span of the wing, i.e. the maximum extent of the wing
                from tip-to-tip.
        """
        # Super class call
        super().__init__()
        self._b = span

        return

    def __getitem__(self, key):
        nd_stations = super().__getitem__(key)

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
        if not isinstance(nd_stations, Hint.iter.__args__):
            nd_stations = [nd_stations]

        # Missing sweep and dihedral angles should be inherited from inboard
        for i, aerofoil in enumerate(nd_stations):
            # If aerofoil is derived, it has parents (one of which is inboard)
            if hasattr(aerofoil, "_parents"):
                parent = self[min(getattr(aerofoil, "_parents"))]
                # Assign private vars, skips float typecasting for 'None' values
                aerofoil._sweep = parent.sweep
                aerofoil._dihedral = parent.dihedral
                delattr(aerofoil, "_parents")

        return collapse1d(nd_stations)


if __name__ == "__main__":
    from carpy.aerodynamics.aerofoil import NewNDAerofoil

    n0012 = NewNDAerofoil.from_procedure.NACA("0012")
    n8412 = NewNDAerofoil.from_procedure.NACA("8412")

    mystations = WingStations(span=30)
    mystations[0] = NDWingStation(n8412)
    mystations[100] = NDWingStation(n0012)
    mystations[60] = mystations[60]

    mystations[:60].sweep = np.radians(0)
    mystations[60:].sweep = np.radians(2)
    print(mystations[0:].sweep)

    mystations[0:].dihedral = np.radians(3)
    print(mystations[:].dihedral)

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
