"""Methods relating to aerofoil profile generation and performance."""
import re

import numpy as np
import requests
from scipy.optimize import minimize
from sectionproperties.analysis.section import Section
from sectionproperties.pre.geometry import Geometry

from carpy.aerodynamics.aerofoil._thinaero import \
    coords2camber, ThinCamberedAerofoil
from carpy.utility import Hint, cast2numpy, isNone, point_curvature, point_diff
from ._generators import (
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


def cast2points(array: Hint.nums):
    """
    Casts an array of 2d coordinates (or points) into points.

    Args:
        array: An array of 2d points and coordinates.

    Returns:
        An array of 2d points of shape (n, 2).

    """
    # Recast as necessary
    array = cast2numpy(array)

    assert array.ndim == 2, f"expected 2d array, got {array.ndim=}"

    if array.shape[1] == 2:
        pass
    elif array.shape[0] == 2:
        array = array.T
    else:
        raise ValueError(f"Expected array of shape (n, 2), got {array.shape}")

    return array


def find_camber(points: np.ndarray, idx_le: int, dxtarget=None) -> np.ndarray:
    """
    Find a set of points that describe the camber line of an aerofoil.

    Args:
        points: Points arranged CCW from TE to LE to TE, describing an aerofoil.
        idx_le: The index of a point in points to be considered the LE.
        dxtarget: A target spacing between camber line designation points.

    Returns:
        An array of points that describe the camber line.

    """
    # Recast as necessary
    dxtarget = 0.075 if dxtarget is None else dxtarget

    # Identify points on the aerofoil to perform camber analysis from
    # ... it's fine to use linear (not cosine!) spacing as the curvature is weak
    upper_points = points[:idx_le + 1]
    camber_xtarget = np.linspace(0, 1, int(1 / dxtarget))
    js = []
    for i in range(1, len(camber_xtarget) - 1):
        # Search upper surface for bounding points, and map camber to it
        for j in range(0, len(upper_points) - 1)[::-1]:
            if 0 < camber_xtarget[i] <= upper_points[j, 0] < 1:
                js.append(j)
                break
    else:
        js = sorted(set(js))  # Indices to use for camber locating

    # Compute normal vectors of points we wish to stage camber circles from
    camber_xy0 = upper_points[js]
    lower_points = points[idx_le:]
    norms = np.ones((len(js), 2))  # Empty array
    norms[:, 1] = -1 / point_diff(*np.fliplr(camber_xy0).T)  # Slopes
    norms[:] = [norm / np.linalg.norm(norm) for norm in norms]  # Normalise
    norms[norms[:, 1] > 0] = -norms[norms[:, 1] > 0]  # Orientation of norm

    # Use the normal vectors to construct camber finding circles
    camber_points = []
    for i, source_point in enumerate(camber_xy0):
        radii = []
        for point in lower_points:
            def f_opt(r1):
                """Opt. func based on radius of inscribed camber circle."""
                camber_point = source_point + r1 * norms[i]
                r2 = np.sum((point - camber_point) ** 2) ** 0.5
                result = abs(r1 - r2)  # difference between radii
                return result

            radius = minimize(
                f_opt, np.zeros(1), bounds=((0, None),), tol=1e-4).x
            radii.append(float(radius))

        camber_points.append(source_point + min(radii) * norms[i])

    # Include the obvious points of leading and trailing edge (0, 0), (1, 0)
    camber_points.insert(0, np.array([1, 0]))
    camber_points.append(np.array([0, 0]))

    return np.array(camber_points[::-1])


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
                aerofoil = Aerofoil(points=naca_class(code).nd_xy(N=N))
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

class Aerofoil(object):
    """
    A class for modelling the non-dimensional geometry and performance of
    aerofoils.
    """

    def __init__(self, *, points: Hint.nums = None,
                 upper_points: Hint.nums = None,
                 lower_points: Hint.nums = None):
        """
        Keyword Args:
            points: A 2d array of points describing the aerofoil profile.
            upper_points: A 2d array of points describing the upper surface
                of the aerofoil.
            lower_points: A 2d array of points describing the upper surface
                of the aerofoil.

        Notes:
            Either the 'points' keyword argument is required to describe the
            aerofoil geometry, or the 'upper_points' and 'lower_points' in
            concert.

        """
        # Recast as necessary (we need to regularise coordinate arguments)
        arg0 = not isNone(points)
        arg12 = not all(isNone(upper_points, lower_points))

        # ... no points are given
        if arg0 is False and arg12 is False:
            errormsg = f"{type(self).__name__}.__init__ requires 'points' or " \
                       f"'upper_points' and 'lower_points' are provided."
            raise ValueError(errormsg)

        # ... geometry is overdefined by points
        elif arg0 is True and arg12 is True:
            errormsg = f"{type(self).__name__}.__init__ received too many " \
                       f"arguments describing the aerofoil's geometry."
            raise ValueError(errormsg)

        # ... __init__ received one array of point coordinates
        elif arg0:
            points = cast2points(points)
            avg_start = points[:(arraylen := len(points)) // 2, 1].mean()
            avg_end = points[arraylen // 2:, 1].mean()
            if avg_start < avg_end:  # points are probably clockwise, cast CCW
                points = points[::-1]

        # ... __init__ received two arrays of point coordinates
        else:
            upper_points, lower_points = \
                map(cast2points, [upper_points, lower_points])
            if np.diff(upper_points[:, 0]).mean() > 0:  # diff suggest clockwise
                upper_points = upper_points[::-1]
            if np.diff(lower_points[:, 0]).mean() < 0:  # diff suggest clockwise
                lower_points = lower_points[::-1]
            # If leading edge points are duplicated, omit before concatenation
            if np.equal(upper_points[-1], lower_points[0]).all():
                points = np.concatenate([upper_points[:-1], lower_points])
            else:
                points = np.concatenate([upper_points, lower_points])
            del upper_points, lower_points

        self._points = points  # This is a counter-clockwise sequence from TE
        self._section = None  # Empty, until its time to compute it via property

        # Determine the coordinates representing the leading edge from curvature
        self._curvature = abs(point_curvature(*points.T))  # Unsigned curvature

        return

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot add {type(self)=} to {type(other)=}")

        # It is not necessarily the case that self and other's upper and lower
        # surfaces share abscissa. For this reason, each surface on each
        # aerofoil must be interpolated.
        idx_le_self = int(np.argmax(self._curvature))
        idx_le_other = int(np.argmax(other._curvature))

        # Determine the number of points which shall define a new surface
        n_points = max(
            len((upper_self := self._points[:idx_le_self + 1][::-1])),
            len((upper_other := other._points[:idx_le_other + 1][::-1])),
            len((lower_self := self._points[idx_le_self:])),
            len((lower_other := other._points[idx_le_other:]))
        )
        xnew = (np.cos(np.linspace(np.pi, 0, n_points)) + 1) / 2
        y_u = np.interp(xnew, *upper_self.T) + np.interp(xnew, *upper_other.T)
        y_l = np.interp(xnew, *lower_self.T) + np.interp(xnew, *lower_other.T)

        new_object = type(self)(
            upper_points=np.vstack([xnew, y_u]).T,
            lower_points=np.vstack([xnew, y_l]).T
        )
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
        # Simply transform the coordinates
        new_object = type(self)(points=self._points * multiplier)
        return new_object

    def __rmul__(self, other):
        return self.__mul__(other)

    @property
    def le_radius(self):
        """Radius of the leading edge."""
        return 1 / self._curvature.max()

    def _camber_points(self, step_target: Hint.num) -> np.ndarray:
        """
        An array of points that describe the aerofoil's camber line.

        Args:
            step_target: Approximate distance between camber designating points.

        Returns:
            A 2D array of points describing the aerofoil's camber line.

        """
        idx_le = int(np.argmax(self._curvature))  # Index of the leading edge
        points = find_camber(
            points=self._points, idx_le=idx_le, dxtarget=step_target)
        return points

    def _thickness_function(self, orientation: Hint.num = None) -> Hint.func:
        # Recast as necessary
        convention = "American" if orientation is None else "British"

        if convention == "American":
            errormsg = (
                "Sorry, this is unavailable right now. You can still proceed "
                "with identifying aerofoil thickness through the British "
                "convention which is possible by passing the 'orientation' arg."
            )
            raise NotImplementedError(errormsg)
        else:
            # Cosine spaced sampling points
            xc = (np.cos(np.linspace(np.pi, 0, 100)) + 1) / 2

            # Translate the leading edge point to (0, 0) (if it isn't already??)
            idx_le = int(np.argmax(self._curvature))  # Index of leading edge
            points = self._points - self._points[idx_le]

            # Transform geometry into the desired orientation for thickness eval
            cos_o, sin_o = np.cos(orientation), np.sin(orientation)
            rot_matrix = np.array([[cos_o, -sin_o], [sin_o, cos_o]])
            points = np.array([rot_matrix @ point for point in points])

            # Compute thickness
            upper_points = points[:idx_le + 1][::-1]  # Order from left to right
            lower_points = points[idx_le:]
            t_hi = np.interp(xc * cos_o, *upper_points.T)
            t_lo = np.interp(xc * cos_o, *lower_points.T)

            def thickness(x: Hint.nums):
                """
                The available thickness of the aerofoil in an orientation
                traditionally perpendicular to the chord line.
                """
                # Recast as necessary
                x = cast2numpy(x)
                return np.interp(x, xc, t_hi - t_lo, left=np.nan, right=np.nan)

            thickness.__doc__ = (
                f"\n    The available thickness of the aerofoil in an"
                f"\n    orientation traditionally perpendicular to the chord"
                f"\n    line. In this case, the available thickness is measured"
                f"\n    in the vertical direction after the aerofoil geometry"
                f"\n    has been rotated counter-clockwise by {orientation:.3f}"
                f" radians."
                f"\n    Args:"
                f"\n    \tx: The chordwise position at which thickness should"
                f"\n    \t\tbe evaluated."
                f"\n"
                f"\n    Returns:"
                f"\n    \tThe thickness of the aerofoil.\n"
            )

        return thickness

    @property
    def section(self) -> Section:
        """Section properties, for geometric analysis of the aerofoil."""
        # If section has been generated previously, reuse it
        if self._section is not None:
            return self._section

        # Create a section...
        # Describe the connectivity of the points in a nested list
        facets = (
                         np.arange(len(self._points))[:, None] +
                         np.array([0, 1])
                 ) % len(self._points)
        facets = [list(x) for x in facets]
        # Identify the inside region of the aerofoil as 10% behind leading edge
        control_points = [[0.1, 0]]

        # Create a geometry object
        section_geometry = Geometry.from_points(
            points=self._points,
            facets=facets,
            control_points=control_points
        )
        section_geometry.create_mesh(mesh_sizes=[1])
        # Use the geometry object to instantiate a Section object
        self._section = Section(section_geometry)
        self._section.calculate_geometric_properties()  # Update geometric props
        return self._section

    def show(self):
        """Simple 2D render of the aerofoil geometry."""
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, dpi=140)
        ax.plot(*self._points.T, c="k")
        ax.plot(*self._camber_points(step_target=0.1).T, c="orange")

        # Make the primary plot pretty
        fig.canvas.manager.set_window_title(f"{self}.show()")
        ax.set_title("Aerofoil 2D Profile")
        ax.set_aspect(1)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.grid(alpha=0.3)

        plt.show()

        return


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
        if not callable(value):
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
            upper_pts, lower_pts = parse_datfile(response.text)
            aerofoil = Aerofoil(upper_points=upper_pts, lower_points=lower_pts)
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

        upper_pts, lower_pts = parse_datfile(filecontents)
        aerofoil = Aerofoil(upper_points=upper_pts, lower_points=lower_pts)

        return aerofoil

    # From procedural generators of aerofoil geometry
    from_procedure = ProceduralProfiles
