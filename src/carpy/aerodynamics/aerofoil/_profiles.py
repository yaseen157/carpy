"""Methods relating to aerofoil profile generation."""
import re

import numpy as np
import requests
from scipy.integrate import simpson as sint_simpson

from carpy.utility import Hint, cast2numpy

__all__ = ["NewNDAerofoil"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Base classes for procedural aerofoil generators
# ---------------------------------------------------------------------------- #
class BaseProfile(object):
    """General parameters of aerofoil profiles."""

    _f_nd_yz: Hint.func = lambda self, x: x * 0
    _f_nd_yt: Hint.func = lambda self, x: x * 0

    def nd_yz(self, x: Hint.nums) -> np.ndarray:
        """
        Non-dimensional camber line (aerofoil mean-line) ordinate.
        """
        # Recast as necessary
        x = cast2numpy(x)
        return self._f_nd_yz(x)

    def nd_yt(self, x: Hint.nums):
        """
        Non-dimensional thickness distribution, as per American convention.
        """
        # Recast as necessary
        x = cast2numpy(x)
        return self._f_nd_yt(x)

    def nd_xy(self, N: int, *, concatenate: bool = None, closeTE: bool = None):
        """
        Non-dimensional coordinates for each of upper and lower surfaces.
        """
        # Recast as necessary
        concatenate = True if concatenate is None else concatenate
        closeTE = False if closeTE is None else closeTE

        # Cosine distribution for higher resolution near the leading edge
        dphi = (np.pi / 2) * (1 / (N - 1))
        xs = 1 - np.cos(np.arange(0, N) * dphi)

        # Gradient of camber line
        eps = 1e-6
        xs_addeps = np.clip(xs + eps, None, 1)
        xs_subeps = np.clip(xs - eps, 0, None)
        dzdx = (self.nd_yz(xs_addeps) - self.nd_yz(xs_subeps)) / (2 * eps)

        # Ordinate rotation angle
        theta = np.arctan(dzdx)

        # Compute upper and lower coordinates
        zs = self.nd_yz(xs)  # Camber line
        ts = self.nd_yt(xs)  # Thickness distribution
        u_abscissa = xs - ts * np.sin(theta)
        u_ordinate = zs + ts * np.cos(theta)
        l_abscissa = xs + ts * np.sin(theta)
        l_ordinate = zs - ts * np.cos(theta)

        # Package nicely and return
        u_coords = np.array([u_abscissa, u_ordinate])
        l_coords = np.array([l_abscissa, l_ordinate])

        if closeTE is True:
            te_coord = np.array([[1], [0]])
            u_coords = np.concatenate([u_coords, te_coord], axis=1)
            l_coords = np.concatenate([l_coords, te_coord], axis=1)

        if concatenate is True:
            # Force CCW traversal of upper surface and omit duplicated LE coord
            u_coords = np.flip(u_coords, axis=1)[:, :-1]
            return np.concatenate([u_coords, l_coords], axis=1)

        return u_coords, l_coords


class BaseNACA(BaseProfile):
    """General methods for NACA aerofoil objects in carpy."""

    _pattern_valid: str = NotImplemented

    def __init__(self, code: str):
        """
        Args:
            code: NACA aerofoil code.
        """
        # Check that the code is valid
        if not self._is_valid(code=code):
            errormsg = f"{type(self).__name__} got an invalid NACA {code=}"
            raise ValueError(errormsg)

        # Create a private object for recording any parsed terms.
        self._parsed = type(
            "Parsed",
            (),
            {
                "LEri": None,  # Leading Edge radius index number
                "c_max_x": None,  # Chordwise location of maximum camber
                "c_max_y": None,  # Maximal camber ordinate
                "cli": None,  # Design section lift coefficient
                "cli_d": None,  # Range of drag bucket above and below cli
                "flag_experimental": None,  # Experimental aerofoils designation
                "flag_reflex": None,  # Boolean, reflexed camber or not
                "meanline_a": None,  # Extent of uniform loading
                "meanline_b": None,  # Extent of non-zero loading
                "meanline_cli": None,  # Design coefficient of lift for meanline
                "p_min_x": None,  # Chordwise location of minimum pressure
                "t_max": None,  # Maximum thickness to chord ratio
                "t_max_x": None,  # Chordwise location of maximum thickness
                "t0_max": None,  # Max thickness of original distribution
            }
        )()

        return

    @classmethod
    def _is_valid(cls, code: str):
        """Verify that an aerofoil code is a valid member of this class."""
        if re.match(cls._pattern_valid, code):
            return True
        return False

    @staticmethod
    def _pop_pattern(string: str, pattern, *args, **kwargs) -> tuple[str, str]:
        """
        Use regular expression matching to find and pop start of a string.

        Args:
            string: The string to be searched.
            pattern: A pattern expected to be found at the start of the string.
            *args: Miscellaneous arguments to pass to re.match().
            **kwargs: Miscellaneous keyword arguments to pass to re.match().

        Returns:
            A tuple containing the original string stripped of any matches, and
                the string matching the pattern specified by the user.

        """
        match = re.match(pattern, string, *args, **kwargs)

        if match is None:
            raise ValueError(f"Start of '{string}' didn't match {pattern=}")

        _, endofmatch = match.span()
        string_sliced = match.string[endofmatch:]  # String with prefix removed
        string_popped = match.group()  # String that matched the desired pattern

        return string_sliced, string_popped


# ============================================================================ #
# Procedural aerofoil generators
# ---------------------------------------------------------------------------- #

class NACA4DigitSeries(BaseNACA):
    """
    References:
        -   NACA Report No.824, Summary of Aerofoil Data
        -   NASA Technical Memorandum 4741

    """
    _pattern_valid = r"^([0]{2}|[1-9]{2})\d{2}$"

    def __init__(self, code: str):
        """
        Args:
            code: NACA aerofoil code.
        """
        # Superclass call
        BaseNACA.__init__(self, code=code)

        # Parse
        code, c_max_y = self._pop_pattern(code, r"\d")
        code, c_max_x = self._pop_pattern(code, r"\d")
        _, t_max = self._pop_pattern(code, r"\d{2}")

        # Recast
        self._parsed.c_max_y = float(c_max_y) / 100
        self._parsed.c_max_x = float(c_max_x) / 10
        self._parsed.t_max = float(t_max) / 100
        return

    def _f_nd_yz(self, x):
        """Private camber line function."""
        # Recast as necessary
        parsed = self._parsed

        # For symmetric aerofoils, this part is easy - camberline is just zero
        if parsed.c_max_y == 0 and parsed.c_max_x == 0:
            return x * 0

        # For cambered aerofoils...
        yz = np.where(
            x <= parsed.c_max_x,
            # Condition True
            parsed.c_max_y * (
                    (2 * parsed.c_max_x - x) * x) / parsed.c_max_x ** 2,
            # Condition False
            parsed.c_max_y * (
                    (1 - 2 * parsed.c_max_x) + 2 * parsed.c_max_x * x - x ** 2)
            / (1 - parsed.c_max_x) ** 2
        )
        return yz

    def _f_nd_yt(self, x):
        """Private thickness distribution function."""
        # Recast as necessary
        parsed = self._parsed

        # Constants of t/c = 0.20 polynomial for thickness
        a = np.array([0.2969, -0.1260, -0.3516, 0.2843, -0.1015])
        exp = np.array([0.5, 1.0, 2.0, 3.0, 4.0])

        # Ordinates for a xx% thickness aerofoil (by normalising t/c=0.20 form)
        def transform(xi):
            """Compute polynomial for thickness distribution."""
            return np.sum(a * xi ** exp) * (parsed.t_max / 0.20)

        yt = np.vectorize(transform)(x)
        return yt


class NACA4DigitModifiedSeries(BaseNACA):
    """
    References:
        -   NACA Report No.824, Summary of Aerofoil Data
        -   NASA Technical Memorandum 4741

    """
    _pattern_valid = r"^([0]{2}|[1-9]{2})\d{2}-\d[2-6]$"

    # noinspection PyProtectedMember
    _f_nd_yz = NACA4DigitSeries._f_nd_yz  # Identical camber function

    def __init__(self, code: str):
        """
        Args:
            code: NACA aerofoil code.
        """
        # Superclass call
        BaseNACA.__init__(self, code=code)

        # Parse
        code, c_max_y = self._pop_pattern(code, r"\d")
        code, c_max_x = self._pop_pattern(code, r"\d")
        code, t_max = self._pop_pattern(code, r"\d{2}")
        code, _ = self._pop_pattern(code, r"-")
        code, LEri = self._pop_pattern(code, r"\d")
        _, t_max_x = self._pop_pattern(code, r"\d")

        # Recast
        self._parsed.c_max_y = float(c_max_y) / 100
        self._parsed.c_max_x = float(c_max_x) / 10
        self._parsed.t_max = float(t_max) / 100
        self._parsed.LEri = int(LEri)
        self._parsed.t_max_x = float(t_max_x) / 10
        return

    def _f_nd_yt(self, x):
        """Private thickness distribution function."""
        # Recast as necessary
        parsed = self._parsed
        t_max_x = parsed.t_max_x

        # Solve linear system to discover missing constants
        # Initialise constants a_i
        a = np.zeros(4)
        exponent = np.array([0.5, 1.0, 2.0, 3.0])
        # Modify a0 as a function of the leading edge radius index
        a0 = 0.2969
        if parsed.LEri != 9:
            a0 *= parsed.LEri / 6
        else:
            a0 *= 3 ** 0.5
        lhs = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [t_max_x ** 0.5, t_max_x, t_max_x ** 2, t_max_x ** 3],
            [0.5 * t_max_x ** -0.5, 1, 2 * t_max_x, 3 * t_max_x ** 2]
        ])
        rhs = np.array([a0, 0, 0.1, 0])
        a = np.linalg.solve(lhs, rhs)

        # Solve linear system to discover missing constants
        # Initialise constants d_i
        d0 = 0.002
        d1 = {
            0.2: 0.200, 0.3: 0.234, 0.4: 0.315, 0.5: 0.465, 0.6: 0.700
        }[t_max_x]
        lhs = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, (1 - t_max_x), (1 - t_max_x) ** 2, (1 - t_max_x) ** 3],
            [0, 1, 2 * (1 - t_max_x), 3 * (1 - t_max_x) ** 2],
        ])
        rhs = np.array([d0, d1, 0.1, 0])
        d = np.linalg.solve(lhs, rhs)

        # Ordinates for a xx% thickness aerofoil (by normalising t/c=0.20 form)
        scalar = (parsed.t_max / 0.20)

        def f_tc_half(xc):
            """Switch thickness distribution being used based on xc position."""
            if xc <= t_max_x:
                result = scalar * np.sum(a * xc ** exponent)
            else:
                result = scalar * np.sum(d * (1 - xc) ** np.arange(4))
            return result

        yt = np.vectorize(f_tc_half)(x)
        return yt


class NACA5DigitSeries(BaseNACA):
    """
    References:
        -   NACA Report No.824, Summary of Aerofoil Data
        -   NASA Technical Memorandum 4741

    """
    _pattern_valid = r"^\d[2-5][01]\d{2}$"

    # noinspection PyProtectedMember
    _f_nd_yt = NACA4DigitSeries._f_nd_yt  # Identical thickness function

    def __init__(self, code: str):
        """
        Args:
            code: NACA aerofoil code.
        """
        # Superclass call
        BaseNACA.__init__(self, code=code)

        # Parse
        code, cli = self._pop_pattern(code, r"\d")
        code, c_max_x = self._pop_pattern(code, r"\d")
        code, flag_reflex = self._pop_pattern(code, r"\d")
        _, t_max = self._pop_pattern(code, r"\d{2}")

        # Recast
        self._parsed.cli = float(cli) * 3 / 20
        self._parsed.c_max_x = float(c_max_x) / 20
        self._parsed.flag_reflex = bool(int(flag_reflex))
        self._parsed.t_max = float(t_max) / 100
        return

    def _f_nd_yz(self, x):
        """Private camber line function."""
        # Recast as necessary
        parsed = self._parsed

        # Check for reflex, else continue as normal
        if parsed.flag_reflex is False:

            # Use look-up to find values of r, k1 that correspond with c_max_x
            i = [0.05, 0.10, 0.15, 0.20, 0.25].index(parsed.c_max_x)
            r = [0.0580, 0.1260, 0.2025, 0.2900, 0.3910][i]
            k1 = [361.400, 51.640, 15.957, 6.643, 3.230][i] * (parsed.cli / 0.3)

            # Camber function
            def f_zc(xc):
                """Camber line with chord position."""
                zc = (k1 / 6) * np.where(
                    xc <= r,
                    xc ** 3 - 3 * r * xc ** 2 + r ** 2 * (3 - r) * xc,
                    r ** 3 * (1 - xc)
                )
                return zc

            yz = np.vectorize(f_zc)(x)

        else:
            # Every single source I've read on this will provide the formulas
            # to generate the constants below (as well as precomputing them) but
            # they never match what I'm able to reproduce by implementing them

            # Use look-up to find  r, k1, n=(k2/k1) that correspond with x_cmax
            i = [0.10, 0.15, 0.20, 0.25].index(parsed.c_max_x)
            r = [0.1300, 0.2170, 0.3180, 0.4410][i]
            k1 = [51.990, 15.793, 6.520, 3.191][i] * (parsed.cli / 0.30)
            n = [0.000764, 0.00677, 0.0303, 0.1455][i]

            # Camber function
            def f_zc(xc):
                """Camber line with chord position."""
                zc = (k1 / 6) * np.where(
                    xc <= r,
                    # Condition True
                    (xc - r) ** 3
                    - n * (1 - r) ** 3 * xc - r ** 3 * xc + r ** 3,
                    # Condition False
                    n * (xc - r) ** 3
                    - n * (1 - r) ** 3 * xc - r ** 3 * xc + r ** 3
                )
                return zc

            yz = np.vectorize(f_zc)(x)

        return yz


class NACA5DigitModifiedSeries(BaseNACA):
    """
    References:
        -   NACA Report No.824, Summary of Aerofoil Data
        -   NASA Technical Memorandum 4741

    """
    _pattern_valid = r"^\d[2-5][01]\d{2}-\d[2-6]$"

    # noinspection PyProtectedMember
    _f_nd_yz = NACA5DigitSeries._f_nd_yz  # Identical camber function
    # noinspection PyProtectedMember
    _f_nd_yt = NACA4DigitModifiedSeries._f_nd_yt  # Identical thickness function

    def __init__(self, code: str):
        """
        Args:
            code: NACA aerofoil code.
        """
        # Superclass call
        BaseNACA.__init__(self, code=code)

        # Parse
        code, cli = self._pop_pattern(code, r"\d")
        code, c_max_x = self._pop_pattern(code, r"\d")
        code, flag_reflex = self._pop_pattern(code, r"\d")
        code, t_max = self._pop_pattern(code, r"\d{2}")
        code, _ = self._pop_pattern(code, r"-")
        code, LEri = self._pop_pattern(code, r"\d")
        _, t_max_x = self._pop_pattern(code, r"\d")

        # Recast
        self._parsed.cli = float(cli) * 3 / 20
        self._parsed.c_max_x = float(c_max_x) / 20
        self._parsed.flag_reflex = bool(int(flag_reflex))
        self._parsed.t_max = float(t_max) / 100
        self._parsed.LEri = int(LEri)
        self._parsed.t_max_x = float(t_max_x) / 10
        return


class NACA6Series(BaseNACA):
    """
    References:
        -   NACA Report No.824, Summary of Aerofoil Data

    """
    _pattern_valid = (
            r"^6\d"  # First two digits
            + r"((,\d)|(\(\d{3}\))|(_{\d})|(_{\(\d{2,3}\)}))?x?-"  # Upto hyphen
            + r"((\d)|(\(\d\.\d+\)))"  # Design lift coefficient
            + r"((\d{2})|(\(\d{2}\.\d+\)))"  # Relative thickness
            + r"(" + (
                "|".join(
                    [  # Suffix
                        "",  # Default mean-line (a = 1.0)
                        r"(,a=0?(\.\d+)?)|(,a=1(\.0+)?)",  # Mean-line only
                        r"(\{(((a=0?(\.\d+)?)|(a=1(\.0+)?)),"  # Mean-line(s)...
                        r"cli=-?\d?(\.\d+).?)+\})"
                        # ...and design lift coefficient(s)
                    ]
                )
            ) + ")$"
    )

    def __init__(self, code: str):
        """
        Args:
            code: NACA aerofoil code.
        """
        # Superclass call
        BaseNACA.__init__(self, code=code)

        # Parse: Series and minimum pressure position
        code, _ = self._pop_pattern(code, r"\d")
        code, p_min_x = self._pop_pattern(code, r"\d")

        # Parse terms before the hyphen
        cli_d = np.nan
        t0_max = None
        if re.match(r"^,\d", code):
            code, _ = self._pop_pattern(code, ",")
            code, cli_d = self._pop_pattern(code, r"\d")
        elif re.match(r"^_{\d}", code):
            code, _ = self._pop_pattern(code, r"_\{")
            code, cli_d = self._pop_pattern(code, r"\d")
            code, _ = self._pop_pattern(code, r"\}")
        elif re.match(r"^\(\d{3}\)", code):
            code, _ = self._pop_pattern(code, r"\(")
            code, cli_d = self._pop_pattern(code, r"\d")
            code, t0_max = self._pop_pattern(code, r"\d{2}")
            code, _ = self._pop_pattern(code, r"\)")
        elif re.match(r"^_{\(\d{3}\)}", code):
            code, _ = self._pop_pattern(code, r"_{\(")
            code, cli_d = self._pop_pattern(code, r"\d")
            code, t0_max = self._pop_pattern(code, r"\d{2}")
            code, _ = self._pop_pattern(code, r"\)}")
        elif re.match(r"^_{\(\d{2}\)}", code):
            code, _ = self._pop_pattern(code, r"_{\(")
            code, t0_max = self._pop_pattern(code, r"\d{2}")
            code, _ = self._pop_pattern(code, r"\)}")

        # Parse the hyphen (and ignore experimental tag)
        code, flag_experimental = self._pop_pattern(code, "^x?")
        code, _ = self._pop_pattern(code, "^-")

        # Parse design lift coefficient and maximum thickness
        if re.match(r"^\d", code):
            code, cli = self._pop_pattern(code, r"\d")
        else:
            code, _ = self._pop_pattern(code, r"\(|")
            code, cli = self._pop_pattern(code, r"\d(\.\d+)?")
            code, _ = self._pop_pattern(code, r"\)|")
        if re.match(r"^\d{2}", code):
            code, t_max = self._pop_pattern(code, r"\d{2}")
        else:
            code, _ = self._pop_pattern(code, r"\(|")
            code, t_max = self._pop_pattern(code, r"\d{2}(\.\d+)?")
            code, _ = self._pop_pattern(code, r"\)|")

        # Parse addenda
        meanline_cli = [float(cli) / 10]
        if re.match("^$", code):
            meanline_a = [1]
        elif re.match(r"^,a=\d?(\.\d+)?$", code):
            code, _ = self._pop_pattern(code, ",a=")
            meanline_a = [code]
        else:
            meanline_a = re.findall(r"a=(\d?(\.\d+)?)", code)
            meanline_cli = re.findall(r"cli=(-?\d?(\.\d+)?)", code)

        # Recast
        self._parsed.p_min_x = float(p_min_x) / 10
        self._parsed.cli_d = float(cli_d) / 10
        self._parsed.t0_max = float(t_max if t0_max is None else t0_max) / 100
        self._parsed.flag_experimental = bool(flag_experimental)
        self._parsed.cli = float(cli) / 10
        self._parsed.t_max = float(t_max) / 100
        self._parsed.meanline_a = np.array(meanline_a, dtype=np.float64)
        self._parsed.meanline_cli = np.array(meanline_cli, dtype=np.float64)
        return

    def _f_nd_yz(self, x):
        """
        Private camber line function.

        Notes:
            According to NACA report No.824, "For special purposes, load
            distributions other than those corresponding to the simple mean
            lines may be obtained by combining two or more types of mean line
            having positive or negative values of the design lift coefficient.
            The geometric and aerodynamic characteristics of such combinations
            may be obtained by algebraic addition of the values for the
            component mean lines". I interpret this as saying, if you have
            multiple mean line designations - the final mean line is simply the
            arithmetic sum of the component mean lines.
        """
        yz_cumulative = np.zeros_like(x)

        for i in range(len(self._parsed.meanline_a)):
            yz_cumulative += self._f_camber(
                cli=self._parsed.meanline_cli[i],
                a=self._parsed.meanline_a[i],
                b=1
            )(x)

        return yz_cumulative

    @classmethod
    def _f_camber(cls, cli: float, a: float = None, b: float = None):
        """Private function to create camberline associated with meanline."""
        # Recast as necessary
        a = a if a is not None else 1.0
        b = b if b is not None else 1.0

        # Unique case when there would otherwise be some crazy singularities
        if a == 1 and b == 1:
            def f_zc(xc):
                """Camber line function."""
                if xc == 0 or xc == 1:
                    return 0.0  # <-- Has to be a float or np.vectorize is sad!!
                zc = -cli / (4 * np.pi) * (
                        (1 - xc) * np.log(1 - xc) + xc * np.log(xc))
                return zc

        # The next case throws an error because it does not make physical sense.
        # If the aerofoil must maintain uniform loading, and then decrease to
        # zero loading by point b - it doesn't make sense for a to be >= b.
        elif a >= b:
            errormsg = (
                f"Extent of uniform loading '{a=}' cannot be greater than or "
                f"equal to the extent of non-zero loading '{b=}'"
            )
            raise ValueError(errormsg)

        # Every other valid case
        else:
            # Unique case when b == 1 would also cause singularities
            if b == 1:
                g = -1 / (1 - a) * (a ** 2 * (0.5 * np.log(a) - 0.25) + 0.25)
                h = 1 / (1 - a) * (
                        0.5 * (1 - a) ** 2 * np.log(1 - a) - 0.25 * (1 - a) ** 2
                ) + g
            else:
                g = -1 / (b - a) * (
                        a ** 2 * (0.5 * np.log(a) - 0.25)
                        - b ** 2 * (0.5 * np.log(b) - 0.25)
                )
                h = 1 / (b - a) * (
                        0.5 * (1 - a) ** 2 * np.log(1 - a)
                        - 0.5 * (1 - b) ** 2 * np.log(1 - b)
                        + 0.25 * (1 - b) ** 2
                        - 0.25 * (1 - a) ** 2
                ) + g

            def f_zc(xc):
                """Camber line function."""
                if xc == 0 or xc == 1:
                    return 0.0  # <-- Has to be float or np.vectorize is sad!!!
                zc = cli / (2 * np.pi * (a + b))
                zc = zc * (1 / (b - a) * (
                        0.5 * (a - xc) ** 2 * np.log(np.abs(a - xc))
                        - 0.5 * (b - xc) ** 2 * np.log(np.abs(b - xc))
                        + 0.25 * (b - xc) ** 2
                        - 0.25 * (a - xc) ** 2
                ) - xc * np.log(xc) + g - h * xc)
                return zc

        return np.vectorize(f_zc)


class NACA16Series(BaseNACA):
    """
    References:
        -   NACA Report No.824, Summary of Aerofoil Data

    """
    _pattern_valid = r"^16-\d{3}((,a=0?(\.\d+)?)|(,a=1(\.0+)?))?$"

    # noinspection PyProtectedMember
    _f_nd_yz = NACA6Series._f_nd_yz  # Identical camber function
    # noinspection PyProtectedMember
    _f_camber = NACA6Series._f_camber
    # noinspection PyProtectedMember
    _f_nd_yt = NACA4DigitModifiedSeries._f_nd_yt  # Identical thickness function

    def __init__(self, code: str):
        """
        Args:
            code: NACA aerofoil code.
        """
        # Superclass call
        BaseNACA.__init__(self, code=code)

        # Parse
        code, _ = self._pop_pattern(code, r"\d")
        code, p_min_x = self._pop_pattern(code, r"\d")
        code, _ = self._pop_pattern(code, "-")
        code, cli = self._pop_pattern(code, r"\d")
        code, t_max = self._pop_pattern(code, r"\d{2}")
        # Meanline
        if re.match("^$", code):
            a = 1
        else:
            code, _ = self._pop_pattern(code, ",a=")
            a = code

        # Recast
        self._parsed.p_min_x = float(p_min_x) / 10
        self._parsed.meanline_cli = [float(cli) / 10]
        self._parsed.t_max = float(t_max) / 100
        self._parsed.meanline_a = [float(a)]
        # Secret definitions, implied by nature of 16-Series
        self._parsed.LEri = 4
        self._parsed.t_max_x = 0.5
        return


class ProceduralProfiles(object):
    """A collection of procedural aerofoil profile generators."""

    @staticmethod
    def NACA(code: str, N=50):
        """
        Create a NACA 4-digit, 4-digit modified, 5-digit, 5-digit modified,
        or 16 series aerofoil.

        Args:
            code: The aerofoil code.
            N: The number of control points on each of upper and lower surfaces.
                Optional, defaults to 50 points on each surface (N=50).

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
                f"series: {naca_classes}"
            )
            raise ValueError(errormsg)


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
            # If line contains 2 numbers, np.array contents and add to group
            if len(matches := re.findall(r"[-+.e\d]+", line)) == 2:
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
    if (surface_u - surface_l)[:, 1].sum() < 0:
        surface_u, surface_l = surface_l, surface_u  # ..., swap if they weren't

    return surface_u, surface_l


# ============================================================================ #
# Public-facing Aerofoil class
# ---------------------------------------------------------------------------- #


class NDAerofoil(object):
    """Non-dimensional Aerofoil object."""

    def __init__(self, upper_points, lower_points):
        self._rawpoints_u = cast2numpy(upper_points)
        self._rawpoints_l = cast2numpy(lower_points)
        return

    @property
    def closedTE(self):
        """True if the trailing edge is closed, False otherwise."""
        if np.array(self._rawpoints_u[-1] == self._rawpoints_l[-1]).all():
            return True
        return False

    def nd_xyCCW(self, closeTE=None) -> np.ndarray:
        """
        Non-dimensional, CCW coordinates of the aerofoil.

        Args:
            closeTE: Flag, set to control if the output coordinates produce a
                closed geometry. Optional, defaults to False.

        Returns:
            np.ndarray: A 2D array of points.

        """
        closeTE = False if closeTE is None else closeTE
        surface_u = self._rawpoints_u
        surface_l = self._rawpoints_l

        # If current TE closure status doesn't match demanded status, match it
        if self.closedTE is False:

            # If need to close the TE
            if closeTE is True:
                te_coord = np.array([[1, 0]])

                # When necessary, add TE point to both of upper/lower surfaces
                if not (te_coord == surface_u[-1]).all():
                    surface_u = np.concatenate([surface_u, te_coord], axis=0)
                if not (te_coord == surface_l[-1]).all():
                    surface_l = np.concatenate([surface_l, te_coord], axis=0)
        else:
            # If need to open the TE
            if closeTE is False:
                surface_u = self._rawpoints_u[:-1]
                surface_l = self._rawpoints_l[:-1]

        # Concatenate CCW, and remove duplicated LE point
        xy = np.concatenate([surface_u[::-1], surface_l[1:]], axis=0)
        return xy

    @property
    def nd_P(self) -> float:
        """Non-dimensional wetted perimeter (scales linearly with chord)."""
        xy = self.nd_xyCCW(closeTE=True).T
        P = (np.sum(np.diff(xy) ** 2, axis=0) ** 0.5).sum()
        return P

    @property
    def nd_S(self) -> float:
        """Non-dimensional cross-section area (scales with chord ** 2)."""
        # Find when x stops decreasing in CCW coords and starts increasing
        xy = self.nd_xyCCW(closeTE=False).T
        dx, _ = np.diff(xy)
        switchpoint = np.argmax(np.isfinite(np.where(dx < 0, np.nan, dx)))

        # Find the area under each curve using Simpson's rule
        xyupper, xylower = xy[:, :switchpoint + 1], xy[:, switchpoint:]
        area_under_upper = sint_simpson(*np.flip(xyupper[::-1], axis=1))
        area_under_lower = sint_simpson(*xylower[::-1])
        S = area_under_upper - area_under_lower
        return S

    def show(self) -> None:
        """Simple 2D render of the aerofoil geometry."""

        # Gather data to resize the primary plot
        ymax = self._rawpoints_u[:, 1].max()
        ymin = self._rawpoints_l[:, 1].min()
        # Gather data to resize the inset plot
        zoom = 25
        dte_u = self._rawpoints_u[-1] - np.array([1, 0])
        dte_l = self._rawpoints_l[-1] - np.array([1, 0])
        inset_ax_radius = ((dte_u ** 2 + dte_l ** 2) ** 0.5).sum() * 2
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
            axes.fill_between(*self.nd_xyCCW().T, 0, alpha=.1, fc="k")
            axes.axhline(y=0, ls="-.", c="k", alpha=0.3)

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
            url: A filepath to a Selig or Lednicer format coordinate data file.

        Returns:
            An Aerofoil object.

        Examples:

            >>> my_path = "https://m-selig.ae.illinois.edu/ads/coord/n0012.dat"
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
