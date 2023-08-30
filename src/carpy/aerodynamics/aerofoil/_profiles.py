"""Methods relating to aerofoil profile generation and performance."""
import re

import numpy as np
import requests
from sectionproperties.analysis.section import Section
from sectionproperties.pre.geometry import Geometry

from carpy.aerodynamics.aerofoil._thinaero import \
    coords2camber, ThinCamberedAerofoil
from carpy.utility import Hint, cast2numpy, isNone
from ._profiles_geometry import (
    NACA4DigitSeries, NACA4DigitModifiedSeries,
    NACA5DigitSeries, NACA5DigitModifiedSeries,
    NACA16Series
)

__all__ = ["NewNDAerofoil", "NDAerofoil"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Support functions
# ---------------------------------------------------------------------------- #


def parse_datfile(coordinates) -> tuple[np.ndarray, np.ndarray]:
    """
    Given information to describe an aerofoil's profile, parse it into two
    arrays (describing upper and lower surface points, respectively). The
    data must be two-dimensional, and contain a point passing through (0, 0).

    Args:
        coordinates: Raw Selig format, raw Lednicer format, tuple of arrays,
            list of arrays, or an array describing aerofoil geometry. The data
            must be two dimensional, with one axis of size two.

    Returns:
        tuple: Two 2D arrays, describing points in upper and lower surfaces of
            the geometry.

    """
    # If instantiated with a string
    if isinstance(coordinates, str):
        parsed_groups = [[]]
        for line in coordinates.splitlines():
            # If line contains any A-z characters, invalidate that line
            if not isNone(re.match(r"[A-z]", line)):
                parsed_groups.append([])
            # If line contains 2 numbers, np.array contents and add to group
            elif len(matches := re.findall(r"[-+.e\d]+", line)) == 2:
                parsed_groups[-1].append(np.array(matches, dtype=float))
            # If line is empty, prepare next grouping
            elif line == "":
                parsed_groups.append([])
        else:
            # No aerofoil is described with 1 coordinate
            coordinates = [np.array(x) for x in parsed_groups if len(x) > 1]

    # If instantiated with an array
    elif isinstance(coordinates, np.ndarray):
        coordinates = [coordinates]

    # If instantiated with a tuple or list of arrays
    elif isinstance(coordinates, (tuple, list)) and all(
            [isinstance(x, np.ndarray) for x in coordinates]):
        pass

    else:
        errormsg = f"Unsupported input type: {type(coordinates)}"
        raise NotImplementedError(errormsg)

    # Sanity check: Coordinates should be 2D with array shape (n, 2)
    for i, array in enumerate(coordinates):
        if (dims := array.ndim) != 2:
            raise ValueError(f"Coordinate array should be 2D (got {dims})")
        if array.shape[0] == 2:
            coordinates[i] = array.T

    # All non-dimensional coordinate descriptions pass through (0, 0)
    # if Selig style (continuous surface)
    if (n_arrays := len(coordinates)) == 1:
        zeroes_idx = np.where(~coordinates[0].any(axis=1))[0][0]
        coordinates = \
            [coordinates[0][:zeroes_idx + 1][::-1], coordinates[0][zeroes_idx:]]
    # if Lednicer style
    elif n_arrays == 2:
        pass
    else:
        errormsg = (
            f"Got {n_arrays} arrays that could describe coordinates when "
            f"only 1 or 2 arrays are expected"
        )
        raise ValueError(errormsg)

    # The upper surface's ordinates average greater than lower surface's...
    surface_u, surface_l = coordinates
    if np.mean(surface_u[:, 1]) < np.mean(surface_l[:, 1]):
        surface_u, surface_l = surface_l, surface_u  # ..., swap if they weren't

    return surface_u, surface_l


# ============================================================================ #
# Private Aerofoil class
# ---------------------------------------------------------------------------- #

class ProceduralProfiles(object):
    """A collection of procedural aerofoil profile generators."""

    @staticmethod
    def NACA(code: str, N=100):
        """
        Create a NACA 4-digit, 4-digit modified, 5-digit, 5-digit modified,
        or 16 series aerofoil.

        Args:
            code: The aerofoil code.
            N: The number of control points on each of upper and lower surfaces.
                Optional, defaults to 100 points on each surface (N=100).

        Returns:
            An Aerofoil object.

        Examples:

            >>> naca_codes = {
            ...     "4-digit": "2412", "4-digit modified": "2412-63",
            ...     "5-digit": "24012", "5-digit modified": "24012-33",
            ...     "16": "16-012", "16 modified": "16-012,a=0.5"
            ... }


            >>> for _, (_, v) in enumerate(naca_codes.items()):
            >>>     ProceduralProfiles.NACA(code=v, N=60).show()

        """
        naca_classes = [
            NACA4DigitSeries, NACA4DigitModifiedSeries,
            NACA5DigitSeries, NACA5DigitModifiedSeries,
            NACA16Series
        ]
        for naca_class in naca_classes:
            # noinspection PyProtectedMember
            if re.match(naca_class._pattern_valid, code):
                aerofoil = \
                    NDAerofoil(*parse_datfile(naca_class(code).nd_xy(N=N).T))
                return aerofoil
        else:
            errormsg = (
                f"{code=} is not a recognised member of any of the following "
                f"series: {[x.__name__ for x in naca_classes]}"
            )
            raise ValueError(errormsg)


# ============================================================================ #
# Public-facing Aerofoil class
# ---------------------------------------------------------------------------- #


class NDAerofoil(object):
    """Non-dimensional Aerofoil object."""

    def __init__(self, upper_points, lower_points):
        # Geometry of the aerofoil
        # Coordinates moving from leading edge to trailing edge
        self._rawpoints_u = cast2numpy(upper_points)
        self._rawpoints_l = cast2numpy(lower_points)
        self._section = None
        # Performance of the aerofoil
        self._theory_thin = ThinCamberedAerofoil(camber_points=self.xz_points)
        self._alpha_zl = None
        self._Clalpha = None
        self._Cl = None
        self._xc_ac = None

        return

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot add {type(self)=} to {type(other)=}")

        # The number of points in self and other surfaces aren't necessarily
        # the same, so some interpolation is required
        n_upper = round((len(self._rawpoints_u) + len(other._rawpoints_u)) / 2)
        n_lower = round((len(self._rawpoints_l) + len(other._rawpoints_l)) / 2)
        xnew_u = (np.cos(np.linspace(np.pi, 0, n_upper)) + 1) / 2
        xnew_l = (np.cos(np.linspace(np.pi, 0, n_lower)) + 1) / 2

        # Create interpolation functions for upper and lower surfaces
        def f_upper(xs):
            """Linear average of the upper surface at each query point in xs."""
            self_interp = np.interp(xs, *self._rawpoints_u.T)
            # noinspection PyProtectedMember
            other_interp = np.interp(xs, *other._rawpoints_u.T)
            return self_interp + other_interp

        def f_lower(xs):
            """Linear average of the lower surface at each query point in xs."""
            self_interp = np.interp(xs, *self._rawpoints_l.T)
            # noinspection PyProtectedMember
            other_interp = np.interp(xs, *other._rawpoints_l.T)
            return self_interp + other_interp

        new_object = type(self)(
            upper_points=np.vstack([xnew_u, f_upper(xs=xnew_u)]).T,
            lower_points=np.vstack([xnew_l, f_lower(xs=xnew_l)]).T
        )
        # Maintain custom zero-lift angles
        new_object.alpha_zl = self.alpha_zl + other.alpha_zl
        return new_object

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        # Typechecking
        if not isinstance(other, Hint.nums.__args__):
            raise TypeError(f"Cannot multiply {type(self)=} by {type(other)=}")
        elif isinstance(other, Hint.num.__args__):
            multiplier = cast2numpy([1, other])
        else:
            multiplier = cast2numpy(other)
        new_object = type(self)(
            upper_points=self._rawpoints_u * multiplier,
            lower_points=self._rawpoints_l * multiplier
        )
        # Maintain custom zero-lift angles
        xscale, yscale = multiplier
        new_object.alpha_zl = (yscale / xscale) * self.alpha_zl
        return new_object

    def __rmul__(self, other):
        return self.__mul__(other)

    @property
    def xz_points(self) -> np.ndarray:
        """Return an array of points approximating the aerofoil's camberline."""
        xz = coords2camber(self._rawpoints_u, self._rawpoints_l)
        return xz

    @property
    def section(self) -> Section:
        """Section properties, for geometric analysis of the aerofoil."""
        # Section exists...
        if self._section is not None:
            return self._section

        # Create a section...
        # Create a list of counter clockwise points, ignore duplicated LE point
        points = list(self._rawpoints_u)[::-1] + list(self._rawpoints_l)[1:]
        # Describe the connectivity of the points in a nested list
        facets = (np.arange((n := len(points)))[:, None] + np.array(
            [0, 1])) % n
        facets = [list(x) for x in facets]
        # Identify the inside region of the aerofoil as 10% behind leading edge
        control_points = [[0.1, 0]]

        # Create a geometry object
        section_geometry = Geometry.from_points(
            points=points,
            facets=facets,
            control_points=control_points
        )
        section_geometry.create_mesh(mesh_sizes=[1])
        # Use the geometry object to instantiate a Section object
        self._section = Section(section_geometry)
        self._section.calculate_geometric_properties()  # Update geometric props
        return self._section

    def show(self) -> None:
        """Simple 2D render of the aerofoil geometry."""

        # Gather data to resize the primary plot
        ymax = self._rawpoints_u[:, 1].max()
        ymin = self._rawpoints_l[:, 1].min()
        # Gather data to resize the inset plot
        zoom = 25
        dte_u = self._rawpoints_u[-1] - np.array([1, 0])
        dte_l = self._rawpoints_l[-1] - np.array([1, 0])
        inset_ax_radius \
            = max(((dte_u ** 2 + dte_l ** 2) ** 0.5).sum() * 2, .005)
        te_xm, te_ym = (dte_u + dte_l) / 2 + np.array([1, 0])

        # Imports
        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset

        # Create primary axes and inset axes
        fig, ax = plt.subplots(1, dpi=140)
        axins = zoomed_inset_axes(parent_axes=ax, zoom=zoom, loc="upper right")

        # Draw on both axes objects
        for axes in (ax, axins):
            axes.plot(*self._rawpoints_u.T, "blue")
            axes.plot(*self._rawpoints_l.T, "gold")
            axes.plot(*self.xz_points.T, "teal", ls="--")
            axes.fill_between(
                *np.array(self.section.geometry.points).T, 0, alpha=.1, fc="k")
            axes.axhline(y=0, ls="-.", c="k", alpha=0.3, lw=1)

        # Make the primary plot pretty
        fig.canvas.manager.set_window_title(f"{self}.show()")
        ax.set_title("Aerofoil 2D Profile")
        ax.set_aspect(1)
        ax.set_xlim(-0.1, 1.16 + zoom * 2 * inset_ax_radius)
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo - (ymax - ymin), yhi + 2 * (ymax - ymin))
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.grid(alpha=0.3)

        # Make the zoomed-in plot pretty
        axins.set_xlim(te_xm - inset_ax_radius, te_xm + inset_ax_radius)
        axins.set_ylim(te_ym - inset_ax_radius, te_ym + inset_ax_radius)
        axins.set_xlabel(f"{zoom}x zoom", fontsize="small")
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=3, fc="none", edgecolor="limegreen")

        plt.show()
        return None

    @property
    def alpha_zl(self) -> float:
        """Angle of attack for zero lift."""
        # Alpha of zero lift is known...
        if self._alpha_zl is not None:
            return self._alpha_zl

        # Else it needs to be computed, update locally stored result
        self._alpha_zl = self._theory_thin.alpha_zl
        return self._alpha_zl

    @alpha_zl.setter
    def alpha_zl(self, value):
        self._alpha_zl = float(value)

    @alpha_zl.deleter
    def alpha_zl(self):
        self._alpha_zl = None

    @property
    def Clalpha(self) -> Hint.func:
        """Function for determining the sectional lift-curve slope."""
        # Lift-curve slope is not known...
        if self._Clalpha is None:
            return self._theory_thin.Clalpha

        return self._Clalpha

    @Clalpha.setter
    def Clalpha(self, value):
        if not isinstance(value, Hint.func.__args__):
            errormsg = (
                f"Clalpha.setter is expecting to be given 'function(alpha)', "
                f"actually got Cla = {value} (invalid {type(value)=})"
            )
            raise TypeError(errormsg)

        self._Clalpha = value
        return

    @Clalpha.deleter
    def Clalpha(self):
        self._Clalpha = None
        return

    @property
    def Cl(self):
        """Function for determining the sectional lift coefficient."""
        # Sectional lift coefficient is not known...
        if self._Cl is None:
            return self._theory_thin.Cl

        return self._Cl

    @Cl.setter
    def Cl(self, value):
        if not isinstance(value, Hint.func.__args__):
            errormsg = (
                f"Cl.setter is expecting to be given 'function(alpha)', "
                f"actually got Cl = {value} (invalid {type(value)=})"
            )
            raise TypeError(errormsg)

        self._Clalpha = value
        return

    @Cl.deleter
    def Cl(self):
        self._Cl = None
        return

    @property
    def xc_ac(self):
        """Chordwise position of the aerofoil's aerodynamic centre."""
        # If the aerodynamic centre is known...
        if self._xc_ac is not None:
            return self._xc_ac

        # Else it must be obtained from theory
        self._xc_ac = self._theory_thin.xc_ac
        return self._xc_ac

    @xc_ac.setter
    def xc_ac(self, value):
        if value > 1 or value < 0:
            raise ValueError(f"xc_ac must be bounded by [0, 1] (got {value})")
        self._xc_ac = float(value)
        return

    @xc_ac.deleter
    def xc_ac(self):
        self._xc_ac = None
        return

    @property
    def xc_cp(self):
        """Chordwise position of the aerofoil's centre of pressure."""
        return self._theory_thin.xc_cp


class NewNDAerofoil(object):
    """
    A class of methods for generating non-dimensional aerofoil geometries.
    """

    # From online sources
    @classmethod
    def from_url(cls, url: str):
        """
        Return an aerofoil object, given the URL to a coordinates data file.

        Args:
            url: A web-URL to a Selig or Lednicer format coordinate data file.

        Returns:
            An Aerofoil object.

        Examples:

            >>> my_url = "https://m-selig.ae.illinois.edu/ads/coord/n0012.dat"
            >>> n0012 = NewNDAerofoil.from_url(url=my_url)
            >>> n0012.show()

        """
        response = requests.get(url=url)

        # On successful request
        if response.status_code == 200:
            geometry = parse_datfile(response.text)
            aerofoil = NDAerofoil(*geometry)
        else:
            raise ConnectionError("Couldn't access given URL")

        return aerofoil

    # From local filepath
    @classmethod
    def from_path(cls, filepath):
        """
        Return an aerofoil object, given the path to a coordinates data file.

        Args:
            filepath: A path to a Selig or Lednicer format coordinate data file.

        Returns:
            An Aerofoil object.

        Examples:

            >>> my_path = "C:\\Users\\Public\\Documents\\naca0012.txt"
            >>> n0012 = NewNDAerofoil.from_path(filepath=my_path)
            >>> n0012.show()

        """
        with open(filepath, "r") as f:
            filecontents = f.read()

        geometry = parse_datfile(filecontents)
        aerofoil = NDAerofoil(*geometry)

        return aerofoil

    # From procedural generators of aerofoil geometry
    from_procedure = ProceduralProfiles
