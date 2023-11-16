"""Code to create aerofoil geometry objects."""

"""Methods relating to aerofoil profile generation and performance."""
from functools import cache
import re

import numpy as np
import requests
from scipy.optimize import minimize
from sectionproperties.analysis.section import Section
from sectionproperties.pre.geometry import Geometry

from carpy.utility import Hint, cast2numpy, isNone, point_curvature, point_diff
from ._aerofoils_generators import (
    NACA4DigitSeries, NACA4DigitModifiedSeries,
    NACA5DigitSeries, NACA5DigitModifiedSeries,
    NACA16Series, ThinParabolic
)

__all__ = ["NewAerofoil", "Aerofoil"]
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
            must be two-dimensional, with one axis of size two.

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

    # Sanity check: Coordinates should be 2D with array shape (2, n)
    for i, array in enumerate(coordinates):
        if (dims := array.ndim) != 2:
            raise ValueError(f"Coordinate array should be 2D (got {dims})")
        if array.shape[0] == 2:
            coordinates[i] = array
        else:
            coordinates[i] = array.T

    # All non-dimensional coordinate descriptions pass through (0, 0)
    # if Selig style (continuous surface)
    if (n_arrays := len(coordinates)) == 1:
        # Look column-wise for any value != 0, then  invert selection to find 0s
        i_0s = int(np.where(~coordinates[0].any(axis=0))[0])
        coordinates = [
            coordinates[0][:, :i_0s + 1][:, :-1],  # Order surface from LE to TE
            coordinates[0][:, i_0s:]  # Order surface from LE to TE
        ]
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
    if np.mean(surface_u[1, :]) < np.mean(surface_l[1, :]):
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
    def NACA(code: str, N: int = None):
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
        # Recast as necessary
        N = 100 if N is None else int(N)

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

    @staticmethod
    def ThinParabolic(epsilon, N=None):
        """
        Produce a zero-thickness aerofoil with parabolic camber.

        Args:
            epsilon: Argument of camber. Zero here would produce an uncambered
                aerofoil.
            N: The number of control points on each of upper and lower surfaces.
                Optional, defaults to 100 points on each surface (N=100).

        Returns:
            An Aerofoil object.

        """
        # Recast as necessary
        N = 100 if N is None else int(N)

        aerofoil = Aerofoil(points=ThinParabolic(epsilon=epsilon).nd_xy(N=N))
        return aerofoil


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
        self._section = None  # Empty until it's time to compute it via property

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

    @property
    def perimeter(self):
        """(Non-dimensional) perimeter ratio to a reference chord of 1."""
        # Take dxs and dys, use pythagoras to find lengths between points, & sum
        return ((np.diff(self._points, axis=0) ** 2).sum(axis=1) ** 0.5).sum()

    def _camber_points(self, step_target: Hint.num,
                       fast: bool = None) -> np.ndarray:
        """
        An array of points that describe the aerofoil's camber line.

        Args:
            step_target: Approximate distance between camber designating points.
            fast: Boolean flag, True if user wishes to use a lower accuracy but
                faster approximation. Optional, defaults to True. See Notes for
                more details.

        Returns:
            A 2D array of points describing the aerofoil's camber line.

        Notes:
            In fast mode, the camber is determined by averaging the coordinates
                of upper and lower surfaces. In slow mode, camber points are
                determined by finding the parameters of circles inscribing the
                aerofoil geometry. This method is not robust, and can have
                trouble - especially with thinner geometry like that found at
                the trailing edge.

        """
        # Recast as necessary
        fast = True if fast is None else fast

        idx_le = int(np.argmax(self._curvature))  # Index of the leading edge

        if fast is True:
            xs = np.arange(0, 1 + step_target, step_target)
            y_u = np.interp(xs, *self._points[:idx_le + 1][::-1].T)
            y_l = np.interp(xs, *self._points[idx_le:].T)
            points = np.column_stack((xs, (y_u + y_l) / 2))
            points[0] = 0, 0
            points[-1] = 1, 0
        else:
            points = find_camber(
                points=self._points, idx_le=idx_le, dxtarget=step_target)

        return points

    def _thickness_function(self, orientation: Hint.num = None) -> Hint.func:
        """
        Create and return a function that estimates the aerofoil thickness at a
        chordwise station in the direction of the given orientation.

        Args:
            orientation: Angle at which the thinkness should be measured.
                Optional, defaults to zero (aerofoil thickness measured as
                per the British convention of camber and thickness definitions).

        Returns:
            The thickness of the aerofoil at the chordwise position and in the
            given orientation.

        """
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
    def points(self) -> np.ndarray:
        """Aerofoil points in an array of shape (n, 2)."""
        return self._points

    @property
    def coords(self) -> np.ndarray:
        """Aerofoil coordinates in an array of shape (2, n)."""
        return self._points.T

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
        ax.plot(*self.coords, c="k")
        camber_points = self._camber_points(step_target=0.05, fast=True)
        ax.plot(*camber_points.T, c="orange")

        # Make the primary plot pretty
        fig.canvas.manager.set_window_title(f"{self}.show()")
        ax.set_title("Aerofoil 2D Profile")
        ax.set_aspect(1)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.grid(alpha=0.3)

        plt.show()

        return


class NewAerofoil(object):
    """
    A class of methods for generating non-dimensional aerofoil geometries.
    """

    # From online sources
    @classmethod
    @cache
    def from_url(cls, url: str):
        """
        Return an aerofoil object, given the URL to a coordinates data file.

        Args:
            url: A web-URL to a Selig or Lednicer format coordinate data file.

        Returns:
            An Aerofoil object.

        Examples:

            >>> my_url = "https://m-selig.ae.illinois.edu/ads/coord/n0012.dat"
            >>> n0012 = NewAerofoil.from_url(url=my_url)
            >>> n0012.show()

        """

        try:
            response = requests.get(url=url)
        except requests.ConnectionError as e:
            errormsg = (
                "Couldn't reach aerofoil geometry database. Is an internet "
                "connection available and working?"
            )
            raise ConnectionError(errormsg) from e

        # On successful request
        if response.status_code == 200:
            upper_pts, lower_pts = parse_datfile(response.text)
            aerofoil = Aerofoil(upper_points=upper_pts, lower_points=lower_pts)
        else:
            errormsg = f"Operation failed with HTTP {response.status_code=}"
            raise ConnectionError(errormsg)

        return aerofoil

    # From local filepath
    @classmethod
    def from_file(cls, filepath):
        """
        Return an aerofoil object, given the path to a coordinates data file.

        Args:
            filepath: A path to a Selig or Lednicer format coordinate data file.

        Returns:
            An Aerofoil object.

        Examples:

            >>> my_path = "C:\\Users\\Public\\Documents\\naca0012.txt"
            >>> n0012 = NewAerofoil.from_file(filepath=my_path)
            >>> n0012.show()

        """
        with open(filepath, "r") as f:
            filecontents = f.read()

        upper_pts, lower_pts = parse_datfile(filecontents)
        aerofoil = Aerofoil(upper_points=upper_pts, lower_points=lower_pts)

        return aerofoil

    # From procedural generators of aerofoil geometry
    from_method = ProceduralProfiles
