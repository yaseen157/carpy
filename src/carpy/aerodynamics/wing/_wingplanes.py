"""Methods for generating and optimising wing planforms."""
import numpy as np

from carpy.aerodynamics.aerofoil import NDAerofoil
from carpy.aerodynamics.wing._pllt import PLLT
from carpy.utility import Hint, cast2numpy

__all__ = ["NDWingStation"]
__author__ = "Yaseen Reza"


class NDWingStation(object):
    """
    Class for modelling the aerofoil sections to attach to a wing.

    A wing station captures the properties of a cross-sectional profile taken of
    the wing perpendicular to the leading edge. This makes station geometry
    independent of sweep, as opposed to wing buttlines (which are cross-sections
    of the wing taken parallel to the aircraft buttock-line).

    Station 0 is often taken as the point where the leading edge of the wing
    meets the fuselage (and thus the station intersects the fuselage).
    """

    def __init__(self, nd_aerofoil: NDAerofoil, theta: Hint.num = None):
        self._nd_aerofoil = nd_aerofoil
        self._theta = theta
        return

    def __add__(self, other):
        # Typechecking
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot add {type(self)=} to {type(other)=}")

        new_object = type(self)(
            nd_aerofoil=self._nd_aerofoil + other._nd_aerofoil,
            theta=self.theta + other.theta
        )
        return new_object

    def __mul__(self, other):
        # Typechecking
        if not isinstance(other, Hint.num.__args__):
            raise TypeError(f"Cannot multiply {type(self)=} by {type(other)=}")
        # New non-dimensional profile and angle of twist
        new_nd_aerofoil = other * self._nd_aerofoil
        new_theta = other * self.theta

        new_object = type(self)(
            nd_aerofoil=new_nd_aerofoil,
            theta=new_theta
        )
        return new_object

    @property
    def theta(self) -> float:
        """
        The station's geometric angle of twist, relative to wing root. A
        positive value indicates wash-in (station has an angle of incidence
        greater than the wing root), and a negative value indicates wash-out.

        Returns:
            Geometric angle of twist of the wing station.

        """
        if self._theta is None:
            self._theta = 0
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value
        return

    @theta.deleter
    def theta(self):
        self._theta = None
        return

    @property
    def alpha_zl(self) -> float:
        """Station's angle of attack for zero-lift, relative to wing root."""
        alpha_zl = self._nd_aerofoil.alpha_zl - self.theta
        return alpha_zl

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

        local_alpha = alpha + self.theta
        Clalpha = self._nd_aerofoil.Clalpha(alpha=local_alpha)
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

        local_alpha = alpha + self.theta
        Cl = self._nd_aerofoil.Cl(alpha=local_alpha)
        return Cl


class WingPlane(object):
    """
    Class for modelling wing planes.
    """

    def __init__(self, span: Hint.num, mirror: bool = None):
        """
        Args:
            span: The full span of the wing, as determined from planform view.
            mirror: Whether or not to mirror the wing's stations about the
                centreline.
        """
        self._b = span
        self._mirror = True if mirror is None else mirror
        self._spar = None

        return

    @property
    def parametric_spar(self) -> Hint.func:
        """
        A parameterised definition of the wingspar's (composite) geometry and
        material selection(s). Accepts arguments for the maximal 'height' and
        'width' of a spar section.

        Returns:
            Spar section object, with geometric properties pre-computed.

        """
        if self._spar is None:
            raise NotImplementedError("Spar definitions has not yet been given")
        return self._spar

    @parametric_spar.setter
    def parametric_spar(self, value):
        if not callable(value):
            errormsg = (
                f"section_spar.setter is expecting to be given 'function(y)', "
                f"actually got section_spar = {value} (invalid {type(value)=})"
            )
            raise TypeError(errormsg)
        self._spar = value
        return

    @parametric_spar.deleter
    def parametric_spar(self):
        self._spar = None
        return


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
