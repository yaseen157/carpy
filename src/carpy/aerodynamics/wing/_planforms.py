"""Methods relating to wing planform generation."""
import numpy as np
from scipy.integrate import quad as sint_quad
from scipy.optimize import minimize as sopt_min

from carpy.utility import Hint, Quantity, cast2numpy

__all__ = ["BasePlanform", "Planforms"]
__author__ = "Yaseen Reza"


class BasePlanform(object):
    """
    General planform parameters (taking 2D birds eye projection).

    References:
        -   DATCOM1978, Section 2.2.2 Planform Parameters

    """
    _b = NotImplemented
    _f_c = NotImplemented
    _f_x = NotImplemented

    def __init__(self):
        errormsg = (
            f"Objects of type '{type(self).__name__}' should not be "
            f"instantiated under normal circumstances, this is the parent of "
            f"child classes you may consider wanting to instantiate from."
        )
        raise EnvironmentError(errormsg)

    def _f_c_setter(self, function):
        """Check to see if port-side chord length < 0, and assign chord func."""
        port_chord_position = np.ones(1) * -self.b.x / 2
        if sopt_min(lambda y: function(y), port_chord_position).fun < 0:
            raise ValueError("Negative chord length encountered in assignment")
        self._f_c = function

    @staticmethod
    def _tan_sweepNN(tan_sweepMM, mm, nn, AR, taper):
        """
        Convert from tangent function of one sweep angle to another, for a
        linearly changing section of wing.

        Args:
            tan_sweepMM: Tangent function of sweep at chordwise station MM.
            mm: Chordwise percentage of sweep station MM.
            nn: Chordwise percentage of target sweep station NN.
            AR: Aspect ratio of wing section.
            taper: Taper ratio of wing section.

        Returns:
            tan_sweepNN, the tangent function of the sweep angle at chordwise
                station NN.

        """
        tan_sweepNN = tan_sweepMM
        tan_sweepNN -= 4 / AR * ((nn - mm) / 100 * (1 - taper) / (1 + taper))
        return tan_sweepNN

    @property
    def AR(self) -> float:
        """Aspect ratio."""
        return (self.b ** 2 / self.S).x

    @property
    def b(self) -> Quantity:
        """Wing span."""
        return self._b

    @property
    def b_2l(self) -> float:
        """Wing-slenderness parameter."""
        return (self.b / 2 / self.l_max).x

    def c(self, y: Hint.nums) -> Quantity:
        """Chord (parallel to axis of symmetry) at span station y."""
        y = cast2numpy(y)
        return self._f_c(y)

    @property
    def cMAC(self) -> Quantity:
        """Mean aerodynamic chord (MAC)."""
        b_2 = self.b / 2
        # noinspection PyTupleAssignmentBalance
        integral, error = sint_quad(lambda y: self.c(y) ** 2, -b_2.x, b_2.x)
        cMAC = 1 / self.S * integral
        return Quantity(cMAC, "m")

    @property
    def cr(self) -> Quantity:
        """Root chord."""
        return self.c(y=0)

    @property
    def l_max(self) -> Quantity:
        """Overall length from wing apex to most aft point on trailing edge."""

        def f_opt(y: float):
            """Helper function: negative vector of length at station y."""
            l_y = self.c(y) + self.x(y)
            return -l_y

        # Find the maximum length from wing apex to aft-most point along TE
        l_max = -sopt_min(f_opt, np.zeros(1)).fun  # invert back to reg. vec.
        return l_max

    @property
    def p(self) -> NotImplemented:
        """Planform-shape parameter."""
        return NotImplemented

    @property
    def S(self) -> Quantity:
        """Wing area."""
        b_2 = self.b / 2
        # noinspection PyTupleAssignmentBalance
        integral, error = sint_quad(self.c, -b_2.x, b_2.x)
        return Quantity(integral, "m^{2}")

    def x(self, y: Hint.nums) -> Quantity:
        """Chordwise location of leading edge at span station y."""
        y = cast2numpy(y)
        return self._f_x(y)

    @property
    def xcentroid(self) -> Quantity:
        """Chordwise location of centroid of area."""

        def integrand(y: float):
            """Helper function: integrand of the centroid function."""
            c = self.c(y)
            x = self.x(y)
            result = c * (x + c / 2)
            return result

        b_2 = self.b / 2
        # noinspection PyTupleAssignmentBalance
        integral, error = sint_quad(integrand, -b_2.x, b_2.x)
        xcentroid = 1 / self.S * integral
        return Quantity(xcentroid, "m")

    @property
    def yMAC(self) -> Quantity:
        """
        Spanwise location of MAC (equivalent to spanwise location of centroid of
        area.
        """
        b_2 = self.b / 2
        # noinspection PyTupleAssignmentBalance
        integral, error = sint_quad(lambda y: self.c(y) * y, -b_2.x, b_2.x)
        yMAC = 1 / self.S * integral
        return Quantity(yMAC, "m")

    def view(self) -> tuple:
        """
        Compute the coordinates of and draw a wing planform in matplotlib.

        Returns:
            A tuple of figure and axes (subplot) objects from matplotlib.

        """
        # Compute coordinates of leading and trailing edges
        wingres = 25
        axres = (wingres * 2) + 1  # Odd points so at least 1 defines centreline
        ys = np.linspace(-0.5, 0.5, axres) * self.b
        xs = self.x(ys)
        cs = self.c(ys)

        # Create a figure and plot things
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(dpi=140)

        # Plot 2d planform
        fb = ax.fill_between(ys, xs, xs + cs, fc="grey", alpha=0.2)
        planform_coords = fb.get_paths()[0].vertices
        ax.plot(*planform_coords.T, c="blue")

        ax.axvline(x=0, c="k", ls="--", alpha=0.3)  # Centreline

        ax.set_title("2D Planform View")
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
        ax.set_ylim(self.l_max * 1.5, -self.l_max / 2)  # Orient axes
        ax.set_aspect(1)  # Ensure scale of x and y axes is comparable/accurate

        return fig, ax


class Trapezoidal(BasePlanform):
    """
    Instances of this class describe the planform geometry of a conventional,
    trapezoidal wing.

    This class can be used to instantiate wing planforms of the following types:

    -   Rectangular (no taper, no sweep)
    -   Straight tapered (tapered, tip chord stays within LE/TE lines of root)
    -   Swept planform (tapered, swept so tip TE or LE is outside of root TE/LE)
    -   Cropped delta (tapered, trailing edge sweep = 0)
    -   Delta (taper ratio = 0, tip chord = 0, LE sweep = 0)
    -   Diamond (taper ratio = 0, tip chord = 0, LE swept back)

    Examples:

        >>> # Create the wing object (constant chord, untapered)
        >>> wing = Trapezoidal(b=11.66, S=13.5)

        >>> # Add taper (taper) and symmetric sweep (sigma = (1 - taper) / 2)
        >>> wing.taper = 0.8
        >>> wing.sigma = 0.1

    References:
        -   DATCOM1978, Section 2.2.2 Planform Parameters

    """

    # noinspection PyMissingConstructor
    def __init__(self, b, S):
        """
        Args:
            b: Projected wing span.
            S: Projected wing area.
        """
        # Set wingspan
        self._b = Quantity(b, "m")

        # Use wing area to set chord (assume x(y) = 0)
        self._f_x = lambda y: np.ones_like(np.array(y)) * Quantity(0, "m")
        self._f_c = lambda y: np.ones_like(np.array(y)) * Quantity(S / b, "m")
        return

    @property
    def cr(self) -> Quantity:
        """
        The chord length of the wingroot. Setting this parameter will
        automatically recompute the wingtip chord length while assuming the
        user desires to maintain leading edge sweep.

        Returns:
            Wingroot chord.

        """
        return self.c(y=0)

    @cr.setter
    def cr(self, value):
        # Simple relations for straight tapered wings for chord root and tip
        cr = float(value)
        ct = (2 * self.b / self.AR / cr) - 1

        b_2 = self.b / 2

        def chord(y):
            """Helper: interpolate chord length between tip and root station."""
            c = np.interp(y, [-b_2.x, 0, b_2.x], [ct.x, cr, ct.x])
            return Quantity(c, "m")

        self._f_c_setter(chord)
        return

    @property
    def ct(self) -> Quantity:
        """
        The chord length of the wingtip. Setting this parameter will
        automatically recompute the wingroot chord length while assuming the
        user desires to maintain leading edge sweep.

        Returns:
            Wingtip chord.

        """
        return self.c(y=-self.b / 2)

    @ct.setter
    def ct(self, value):
        # Simple relations for straight tapered wings for chord root and tip
        cr = 2 * self.b / self.AR - value
        ct = float(value)

        b_2 = self.b / 2

        def chord(y):
            """Helper: interpolate chord length between tip and root station."""
            c = np.interp(y, [-b_2.x, 0, b_2.x], [ct, cr.x, ct])
            return Quantity(c, "m")

        self._f_c_setter(chord)
        return

    def set_sweep(self, xx: Hint.num, /, sweep: Hint.num = None) -> None:
        """
        Define a constant sweep across each semi-span of the wing. The wing
        geometry while maintaining the taper ratio.

        Args:
            xx: Chord wise station in percent at which to set sweep angle.
            sweep: The angle of the sweep at the given chord wise station.
                Optional, defaults to 0 (unswept).

        Returns:
            None

        """
        # Recast as necessary
        xx = float(xx)
        sweep = float(0 if sweep is None else sweep)

        # Compute tangent function of leading edge sweep
        tan_sweepxx = np.tan(sweep)
        tan_sweepLE = self._tan_sweepNN(
            tan_sweepMM=tan_sweepxx, mm=xx, nn=0, AR=self.AR, taper=self.taper)

        # Set sweep using sigma
        self.sigma = self.AR / 4 * (1 + self.taper) * tan_sweepLE

        return None

    @property
    def sweepLE(self) -> float:
        """Leading edge sweep angle."""
        b_2 = self.b / 2
        xt = self.x(y=-b_2)
        xr = self.x(y=0)

        sweepLE = np.arctan2((xt.x - xr.x), b_2.x)

        return sweepLE

    @property
    def sweepTE(self) -> float:
        """Trailing edge sweep angle."""
        b_2 = self.b / 2
        xt, ct = self.x(y=-b_2), self.ct
        xr, cr = self.x(y=0), self.cr

        sweepTE = np.arctan2(((xt + ct).x - (xr + cr).x), b_2.x)

        return sweepTE

    def sweepXX(self, xx: Hint.nums, /) -> np.ndarray:
        """
        Return the sweep angle of the wing at a given chordwise station.

        Args:
            xx: Chordwise position in percent.

        Returns:
            Sweep angle of the wing at the chord position specified.

        """
        # Recast as necessary
        xx = cast2numpy(xx)

        tan_sweepLE = np.tan(self.sweepLE)
        tan_sweepxx = self._tan_sweepNN(
            tan_sweepMM=tan_sweepLE, mm=0, nn=xx, AR=self.AR, taper=self.taper
        )
        return np.arctan(tan_sweepxx)

    @property
    def taper(self) -> float:
        """
        Wing taper ratio, lambda = ct/cr. Setting this parameter automatically
        compute appropriate wingroot and wingtip chords  while assuming the
        user desires to maintain leading edge sweep.

        Returns:
            Wing taper ratio.

        """
        return (self.ct / self.cr).x

    @taper.setter
    def taper(self, value):
        # Simple relations for straight tapered wings for chord root and tip
        cr = 2 * self.b / self.AR / (1 + value)
        ct = cr * value

        b_2 = self.b / 2

        def chord(y):
            """Helper: interpolate chord length between tip and root station."""
            c = np.interp(y, [-b_2.x, 0, b_2.x], [ct.x, cr.x, ct.x])
            return Quantity(c, "m")

        self._f_c_setter(chord)
        return

    @property
    def sigma(self) -> float:
        """
        Parameter of wing sweep, the fraction of the wingroot chord behind the
        wingroot leading edge at which the wingtip leading edge should start.

        Returns:
            Ratio of chordwise position of leading edge at wingtip to reference
                start position of wingroot chord.

        """
        return (self.x(y=-self.b / 2) / self.cr).x

    @sigma.setter
    def sigma(self, value):
        xt = self.cr * value

        b_2 = self.b / 2

        def xlocate(y):
            """Helper: interpolate chordwise location of leading edge at y."""
            x = np.interp(y, [-b_2.x, 0, b_2.x], [xt.x, 0, xt.x])
            return Quantity(x, "m")

        self._f_x = xlocate
        return


class Cranked(BasePlanform):
    """
    Instances of this class describe the planform geometry of cranked wings.

    Types of cranked wing planform include:

    -   Compound tapered (variable chord inboard/outboard panels)
    -   Semi tapered (constant chord inboard panels)
    -   Crescent (inboard sweep angles exceed outboard sweep angles)
    -   Double-delta (cropped delta with compound LE sweep)
    -   Cranked arrow (like crescent, except inboard TE is swept forward)
    -   M-wing (inboard swept forward, outboard swept aft)
    -   W-wing (inboard swept aft, outboard swept forward)

    Examples:

        >>> # Create the wing object (constant chord, untapered, break @3metres)
        >>> wing = Cranked(b=11.66, S=33.5, yB=3)

        >>> # Produce cranked arrow wing planform shape
        >>> wing.sweepLE_io = np.radians(45), np.radians(22)
        >>> wing.sweepTE_io = -np.radians(12), np.radians(6)

    References:
        -   DATCOM1978, Section 2.2.2 Planform Parameters

    """

    # noinspection PyMissingConstructor
    def __init__(self, b, S, yB):
        """
        Args:
            b: Projected wing span.
            S: Projected wing area.
            yB: Projected distance from centreline where wing cranks ("breaks").
        """
        # Set wingspan
        self._b = Quantity(b, "m")

        # Use wing area to set chord (assume x(y) = 0)
        self._f_x = lambda y: np.ones_like(np.array(y)) * Quantity(0, "m")
        self._f_c = lambda y: np.ones_like(np.array(y)) * Quantity(S / b, "m")

        # Initial value
        self._yB = Quantity(yB, "m")
        return

    @property
    def b_io(self) -> tuple[Quantity, Quantity]:
        """
        Span of planform formed by two (inboard, outboard) panels as if they
        were joined together as an independent, isolated wing.
        """
        b_i = 2 * self.yB
        return b_i, self.b - b_i

    @property
    def cB(self) -> Quantity:
        """Chord at break (kink) span station."""
        return self.c(y=self.yB)

    @property
    def ct(self) -> Quantity:
        """Tip chord."""
        return self.c(y=-self.b / 2)

    @property
    def S_io(self) -> tuple[Quantity, Quantity]:
        """Total projected area of (inboard, outboard) panels."""
        S_i = 2 * ((self.cr + self.cB) / 2 * self.yB)
        return S_i, self.S - S_i

    def set_sweep(self, xx: Hint.num, /, sweep: Hint.num = None) -> None:
        """
        Define a constant sweep across each semi-span of the wing, while
        maintaining the taper ratios of the original wing definition.

        Args:
            xx: Chord wise station in percent at which to set sweep angle.
            sweep: The angle of the sweep at the given chord wise station.
                Optional, defaults to 0 (unswept).

        Returns:
            None

        """
        # Recast as necessary
        xx = float(xx)
        sweep = float(0 if sweep is None else sweep)

        # Constant taper implies constant wing panel areas, get aspect ratio
        b_i, b_o = self.b_io
        S_i, S_o = self.S_io
        AR_i = (b_i ** 2 / S_i).x
        AR_o = (b_o ** 2 / S_o).x

        # Compute new LE sweeps
        taper_i, taper_o = self.taper_io
        tan_sweepxx = np.tan(sweep)
        tan_sweepLE_i = self._tan_sweepNN(
            tan_sweepMM=tan_sweepxx, mm=xx, nn=0, AR=AR_i, taper=taper_i)
        tan_sweepLE_o = self._tan_sweepNN(
            tan_sweepMM=tan_sweepxx, mm=xx, nn=0, AR=AR_o, taper=taper_o)

        self.sweepLE_io = np.arctan([tan_sweepLE_i, tan_sweepLE_o])
        return None

    @property
    def sweepLE_io(self) -> tuple[np.float64, np.float64]:
        """
        The inboard and outboard panel leading edge sweeps. Setting this
        parameter with a tuple will change the wing geometry while maintaining
        the taper ratios.

        Returns:
            Tuple of (inboard, outboard) panel leading edge sweeps.

        """
        b_2 = self.b / 2
        xt = self.x(y=-b_2)
        xB = self.x(y=-self.yB)
        xr = self.x(y=0)

        sweepLE_i = np.arctan2((xB.x - xr.x), self.yB.x)
        sweepLE_o = np.arctan2((xt.x - xB.x), (b_2.x - self.yB.x))

        return sweepLE_i, sweepLE_o

    @sweepLE_io.setter
    def sweepLE_io(self, value):
        # Unpack value(s)
        sweepLE_i, sweepLE_o = value

        # Compute start positions of break span and tip
        b_2 = self.b / 2
        xB = self.yB * np.tan(sweepLE_i)
        xt = (b_2 - self.yB) * np.tan(sweepLE_o) + xB
        ys = [-b_2.x, -self.yB.x, 0, self.yB.x, b_2.x]
        xs = [xt.x, xB.x, 0, xB.x, xt.x]

        def xlocate(y):
            """Helper: interpolate chordwise location of leading edge at y."""
            x = np.interp(y, ys, xs)
            return Quantity(x, "m")

        self._f_x = xlocate
        return

    @property
    def sweepTE_io(self) -> tuple[np.float64, np.float64]:
        """Trailing edge sweep of inboard and outboard panels."""
        ys = Quantity([-self.b.x / 2, - self.yB.x, 0], "m")
        xs = self.x(y=ys)
        cs = self.c(y=ys)

        sweepTE_o, sweepTE_i = np.arctan2(-np.diff(xs + cs), np.diff(ys)).x
        return sweepTE_i, sweepTE_o

    @sweepTE_io.setter
    def sweepTE_io(self, value):
        # Unpack value(s)
        sweepLE_i, sweepLE_o = self.sweepLE_io
        sweepTE_i, sweepTE_o = value

        b_2 = self.b / 2
        dy_o = b_2 - self.yB

        # noinspection PyShadowingNames
        def f_crct(cB):
            """For a given cB, compute the cr and ct for the given sweeps."""
            cr = cB + self.yB * np.tan(sweepLE_i) - self.yB * np.tan(sweepTE_i)
            ct = cB - dy_o * np.tan(sweepLE_o) + dy_o * np.tan(sweepTE_o)
            return cr, ct

        # noinspection PyShadowingNames
        def f_opt(cB):
            """For a given cB, find the delta between predicted/actual area."""
            cr, ct = f_crct(cB=cB)
            Si = 2 * ((cr + cB) / 2 * self.yB)
            So = 2 * ((cB + ct) / 2 * dy_o)
            return abs(self.S.x - Si.x - So.x)

        cB = Quantity(sopt_min(f_opt, self.c(y=-self.b / 4)).x, "m")
        cr, ct = f_crct(cB=cB)
        ys = [-b_2.x, -self.yB.x, 0, self.yB.x, b_2.x]

        def chord(y):
            """Helper: interpolate chord length between tip and root station."""
            c = np.interp(y, ys, [ct.x, cB.x, cr.x, cB.x, ct.x])
            return Quantity(c, "m")

        self._f_c_setter(chord)
        return

    def sweepXX(self, xx: Hint.nums, /) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the sweep angle of the wing at a given chordwise station.

        Args:
            xx: Chordwise position in percent.

        Returns:
            Sweep angle of the wing at the chord position specified.

        """
        # Recast as necessary
        xx = cast2numpy(xx)
        sweepLE_i, sweepLE_o = self.sweepLE_io
        taper_i, taper_o = self.taper_io
        b_i, b_o = self.b_io
        S_i, S_o = self.S_io
        AR_i, AR_o = (b_i ** 2 / S_i).x, (b_o ** 2 / S_o).x

        # Compute inboard and outboard sweeps at station
        tan_sweepLE_i = np.tan(sweepLE_i)
        tan_sweepxx_i = self._tan_sweepNN(
            tan_sweepMM=tan_sweepLE_i, mm=0, nn=xx, AR=AR_i, taper=taper_i
        )
        tan_sweepLE_o = np.tan(sweepLE_o)
        tan_sweepxx_o = self._tan_sweepNN(
            tan_sweepMM=tan_sweepLE_o, mm=0, nn=xx, AR=AR_o, taper=taper_o
        )
        return np.arctan(tan_sweepxx_i), np.arctan(tan_sweepxx_o)

    @property
    def taper_io(self) -> tuple[float, float]:
        """Taper ratio of inboard wing panel, taper ratio = cB / cr."""
        return (self.cB / self.cr).x, (self.ct / self.cB).x

    @taper_io.setter
    def taper_io(self, value):
        # Unpack value(s)
        taper_i, taper_o = value

        b_2 = self.b / 2
        dy_o = b_2 - self.yB

        # noinspection PyShadowingNames
        def f_crct(cB):
            """For a given cB, compute the cr and ct for the given tapers."""
            cr = cB / taper_i
            ct = cB * taper_o
            return cr, ct

        # noinspection PyShadowingNames
        def f_opt(cB):
            """For a given cB, find the delta between predicted/actual area."""
            cr, ct = f_crct(cB=cB)
            Si = 2 * ((cr + cB) / 2 * self.yB)
            So = 2 * ((cB + ct) / 2 * dy_o)
            return abs(self.S.x - Si.x - So.x)

        cB = Quantity(sopt_min(f_opt, self.c(y=-self.b / 4)).x, "m")
        cr, ct = f_crct(cB=cB)
        ys = [-b_2.x, -self.yB.x, 0, self.yB.x, b_2.x]

        def chord(y):
            """Helper: interpolate chord length between tip and root station."""
            c = np.interp(y, ys, [ct.x, cB.x, cr.x, cB.x, ct.x])
            return Quantity(c, "m")

        self._f_c_setter(chord)
        return

    @property
    def yB(self) -> Quantity:
        """Spanwise location of break (kink) span station."""
        return self._yB

    @property
    def etaB(self) -> float:
        """Ratio of spanwise location of break (kink) station to semi-span."""
        return (self.yB / (self.b / 2)).x


class Planforms(object):
    """A collection of wing planforms, packaged for easy access."""
    Trapezoidal = Trapezoidal
    Cranked = Cranked
