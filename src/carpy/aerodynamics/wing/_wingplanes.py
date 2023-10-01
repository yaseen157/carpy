"""Methods for generating and optimising wing planforms."""
import numpy as np

from carpy.aerodynamics.aerofoil import Aerofoil
from carpy.structures import DiscreteIndex
from carpy.utility import Hint, Quantity, collapse_array, isNone

__all__ = ["WingSection", "WingSections"]
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


class WingSections(DiscreteIndex):

    def __init__(self):
        # Super class call
        super().__init__()

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


if __name__ == "__main__":
    from carpy.aerodynamics.aerofoil import NewAerofoil

    n0012 = NewAerofoil.from_method.NACA("0012")
    n8412 = NewAerofoil.from_method.NACA("8412")

    # Define buttock-line geometry
    mysections = WingSections()
    mysections[0] = WingSection(n8412)
    mysections[60] = mysections[0].deepcopy()
    mysections[100] = WingSection(n0012)

    # Add leading edge sweep and dihedral
    mysections[:60].sweep = np.radians(0)
    mysections[60:].sweep = np.radians(2)
    mysections[0:].dihedral = np.radians(3)

    # Introduce wing taper
    mysections[0:].chord = 1.0
    mysections[100].chord = 0.4

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

if __name__ == "__main__":
    from sectionproperties.pre.library import steel_sections, primitive_sections
    from sectionproperties.pre.pre import Material
    from sectionproperties.analysis.section import Section

    steel = Material(
        name='Steel', elastic_modulus=200e9, poissons_ratio=0.3,
        density=7.85e3, yield_strength=500e6, color='grey'
    )
    timber = Material(
        name='Timber', elastic_modulus=8e9, poissons_ratio=0.35,
        density=6.5e2, yield_strength=20e6, color='burlywood'
    )


    def sections(height: float, width: float):
        """

        Args:
            height: Maximum allowable height of the spar.
            width: Maximum allowable width of the spar.

        Returns:

        """
        # Compute parameterised dimensions
        core_y = height - (2 * width)
        tube_od = width
        tube_wt = 1e-3

        # STEP 1: Create component section geometries (and attach materials)
        rod_kwargs = {"d": tube_od, "t": tube_wt, "n": 100, "material": steel}
        core_kwargs = {"b": rod_kwargs["d"], "d": core_y, "material": timber}
        rod = steel_sections.circular_hollow_section(**rod_kwargs)
        core = primitive_sections.rectangular_section(**core_kwargs)

        # STEP 2: Create a compound geometry, and convert into a section object
        section_geometry = (
                rod.shift_section(0, (core_y + rod_kwargs["d"]) / 2)
                + core.shift_section(-rod_kwargs["d"] / 2, -core_y / 2)
                + rod.shift_section(0, -(core_y + rod_kwargs["d"]) / 2)
        )
        section_geometry.create_mesh(mesh_sizes=[1e-3])
        section = Section(section_geometry)

        # STEP 3: Calculate the geometric properties of the section
        section.calculate_geometric_properties()

        return section


    section = sections(height=0.1, width=12e-3)
    section.plot_mesh()
