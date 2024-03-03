"""Methods relating to aerofoil profile generation."""
import re

import numpy as np

from carpy.utility import Hint, cast2numpy

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
        # zero loading by point b - it doesn't make sense for 'a' to be >= 'b'.
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


class ThinParabolic(BaseProfile):
    def __init__(self, epsilon: Hint.num):
        self._epsilon = epsilon
        return

    def _f_nd_yz(self, x):
        """Private camber line function."""
        yz = 4 * self._epsilon * x * (1 - x)
        return yz

    @staticmethod
    def _f_nd_yt(x):
        """Private thickness distribution function."""
        return np.ones_like(x) * 1e-3
