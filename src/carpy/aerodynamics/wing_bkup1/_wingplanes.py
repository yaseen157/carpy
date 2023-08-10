"""Methods for generating wing planes."""
import warnings

import numpy as np
from scipy.integrate import trapezoid
import scipy.optimize as sopt

from carpy.utility import cast2numpy, Hint

__all__ = ["NewNDWing"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Support functions
# ---------------------------------------------------------------------------- #


# ============================================================================ #
# Support classes
# ---------------------------------------------------------------------------- #

class WingStation(object):
    """
    Class for storing information pertaining to the shape of the wing (aerofoil)
    and its orientation in a larger wing structure.
    """

    def __init__(self, nd_profile, alpha_geo=None):
        """
        Args:
            nd_profile: A non-dimensional aerofoil profile geometry object.
            alpha_geo: The angle of geometric twist with respect to the root
                chord of the wing.
        """
        self._nd_profile = nd_profile
        self._alpha_geo = 0 if alpha_geo is None else alpha_geo
        return

    def __add__(self, other):
        # Typechecking
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot add {type(self)=} to {type(other)=}")

        new_object = type(self)(
            nd_profile=self.nd_profile + other.nd_profile,
            alpha_geo=self.alpha_geo + other.alpha_geo
        )
        return new_object

    def __mul__(self, other):
        # Typechecking
        if not isinstance(other, Hint.num.__args__):
            raise TypeError(f"Cannot multiply {type(self)=} by {type(other)=}")
        # New non-dimensional profile and angle of twist
        new_nd_profile = other * self._nd_profile
        new_alpha_geo = other * self._alpha_geo

        new_object = type(self)(
            nd_profile=new_nd_profile,
            alpha_geo=new_alpha_geo
        )
        return new_object

    def __rmul__(self, other):
        return self.__mul__(other)

    @property
    def nd_profile(self):
        """
        The non-dimensional aerofoil object attached to this wing station.

        Returns:
            NDAerofoil object.

        """
        return self._nd_profile

    @property
    def CLalpha(self):
        """Lift slope function from the non-dimensional profile."""
        return self.nd_profile.Clalpha

    @property
    def alpha_zl(self):
        """Angle of zero-lift from the non-dimensional profile."""
        return self.nd_profile.alpha_zl

    @property
    def alpha_geo(self) -> float:
        """
        A property of the non-dimensional aerofoil profile attached to this
        station, the angle of geometric twist with respect to the wing root
        aerofoil's reference angle of zero incidence.

        Returns:
            Angle of geometric twist, in radians.

        """
        return self._alpha_geo

    @alpha_geo.setter
    def alpha_geo(self, value):
        self._alpha_geo = value
        return


# ============================================================================ #
# Public-facing Wing class
# ---------------------------------------------------------------------------- #

class NDWing(object):
    """Non-dimensional Wing object."""

    _stations: dict[Hint.num, WingStation]

    def __init__(self, mirror=None):
        """
        Args:
            mirror: Set to True if stations should be mirrored about the vehicle
                centreline. Set to False if asymmetry is required (e.g. defining
                the geometry of a single propeller blade, or aircraft rudders).

        """
        self._mirror = True if mirror is None else mirror
        self._stations = dict()
        self._ctrlpts = (np.array([-1, 0, 1]), np.array([1, 1, 1]))

        return

    @property
    def mirror(self):
        """Whether or not the wing is mirrored about the vehicle centreline."""
        return self._mirror

    @mirror.setter
    def mirror(self, value):
        self._mirror = value
        return

    @mirror.deleter
    def mirror(self):
        # Restore defaults
        self._mirror = True
        return

    @property
    def nd_controlpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Description of the distribution of chord in the wing. This is *not* the
        same as the wing's planform.

        The distribution is described by two arrays - the first of which depicts
        spanwise position, and the latter informs the relative chord length at
        the spanwise position. Both arrays are  non-dimensionalised with respect
        to a reference chord length of one (usually the chord at the vehicle
        centreline/longitudinal axis). When mirrored, the first array looks like
        [-(b/2)/c_ref, ..., 0, ..., (b/2)/c_ref] where c_ref=1, and the second
        array has the form [lambda, ..., 1, ..., lambda] where lambda=taper.

        Returns:
            tuple: A two-element tuple consisting of arrays that describe the
                distribution of chord in the wing.

        """
        if self.mirror is True:
            return self._ctrlpts
        else:
            n = int(len(self._ctrlpts[0]) / 2)
            return self._ctrlpts[0][n:], self._ctrlpts[1][n:]

    def new_station(self, y: float, nd_profile, alpha_geo=None) -> None:
        """
        Add a new station to the definition of this non-dimensional wing
        geometry.

        Args:
            y: A float bounded by [0, 1] (such that 0 <= y <= 1), describing
                the position along the semi-span this station is positioned.
                A value of y=0 indicates the station at the vehicle centreline.
            nd_profile: A non-dimensional aerofoil profile geometry object.
            alpha_geo: The angle of geometric twist with respect to the root
                aerofoil's reference angle of zero incidence.

        Returns:
            None.

        """
        # Check that the non-dimensional station is a valid definition
        if not (0 <= (key := float(y)) <= 1):
            raise ValueError(f"'y' should be bounded to [0, 1] (got {y=})")

        new_station = WingStation(nd_profile=nd_profile, alpha_geo=alpha_geo)
        self._stations[key] = new_station
        return None

    def interp_station(self, y: Hint.nums) -> np.ndarray[WingStation]:
        """
        For non-dimensional station position y (element of [0, 1]), interpolate
        reference station coordinates to produce intermediary stations.

        Args:
            y: Numbers from 0 (wing root) to 1 (wing tip) at which stations are
                requested.

        Returns:
            np.ndarray: WingStation objects at the requested spans.

        """
        # Recast as necessary
        y = cast2numpy(y)

        # Check that the non-dimensional station is a valid definition
        if (~((0 <= y) & (y <= 1))).any():
            raise ValueError(f"'y' should be bounded to [0, 1] (got {y=})")

        # Unpack available stations
        if len(self._stations) == 0:
            errormsg = (
                f"No stations exist to interpolate between. Consider adding a "
                f"new station with the .{self.new_station.__name__}() method"
            )
            raise ValueError(errormsg)

        station_ys, station_geoms = zip(*self._stations.items())
        station_ys, station_geoms = map(cast2numpy, [station_ys, station_geoms])

        # Pad missing stations if necessary
        if not any(station_ys == 0):
            new_station = self._stations[min(station_ys)]
            station_ys = np.concatenate([[0], station_ys])
            station_geoms = np.concatenate([[new_station], station_geoms])

        if not any(station_ys == 1):
            new_station = self._stations[max(station_ys)]
            station_ys = np.concatenate([station_ys, [1]])
            station_geoms = np.concatenate([station_geoms, [new_station]])

        # Sort stations, ready for interpolation
        sort_idxs = np.argsort(station_ys)
        station_ys = station_ys[sort_idxs]
        station_geoms = station_geoms[sort_idxs]
        int_stations = np.empty_like(y, dtype=object)
        idx_stations = np.arange(len(station_ys))

        for i, yi in enumerate(y.flat):
            # Identify bounding indices
            i_lo = (idx_stations[station_ys <= yi]).max()
            i_hi = (idx_stations[yi <= station_ys]).min()

            # If an aerofoil lands exactly on a defined station, copy it exactly
            if i_lo == i_hi:
                # Multiply by 1.0 to forcibly create a new, identical object
                int_stations[i] = station_geoms[i_lo] * 1.0
                continue

            # Otherwise, compute linearly interpolating geometry
            w = (yi - station_ys[i_lo]) / (station_ys[i_hi] - station_ys[i_lo])
            weights = np.array([w, (1 - w)])
            int_stations[i] = (weights * station_geoms[[i_hi, i_lo]]).sum()

        return int_stations

    def _compute_gld(
            self, f_nd_chord: Hint.func, AR: float, alpha_inf: Hint.nums = None,
            N: int = None) -> np.ndarray:
        """
        Compute the decomposed Fourier series solution to the general planform
        circulation distribution problem in Prandtl's Lifting Line Theory.

        Args:
            f_nd_chord: A function with domain [-1, 1] (indicating the fraction
                of semi-span from port wingtip to starboard wingtip), and range
                [0, 1] (indicating a fraction of some reference chord length,
                which is usually the wing root chord).
            AR: Design aspect ratio of the wing.
            alpha_inf: The freestream angle of attack.
            N: The number of sine-spaced samples to compute over the span of the
                [-1, 1] wing domain. Optional, defaults to 50.

        Returns:
            np.ndarray: An array with the same outer dimensions as alpha_inf,
                and deepest inner dimension is the Fourier series solution.

        """
        # Recast as necessary
        alpha_inf = np.zeros(1) if alpha_inf is None else cast2numpy(alpha_inf)
        N = 100 if N is None else N  # ~3 s.f. precision

        theta0 = np.linspace(0, np.pi, N + 2)[1:-1]  # Arguments of distribution

        # Setup some arrays for computation
        interp_stations = self.interp_station(y=np.abs(np.cos(theta0)))
        chord = f_nd_chord(np.cos(theta0))
        alpha_geo = cast2numpy([[x.alpha_geo] for x in interp_stations])
        alpha_zl = cast2numpy([[x.alpha_zl] for x in interp_stations])
        f_clalpha_2d = [x.Clalpha for x in interp_stations]

        # Since S = b * Standard.Mean.Chord; S / (b/2) == 2 * (S / b) == 2 * SMC
        # And now b = AR * SMC, the span we need for a fixed aspect ratio
        twoS_b = abs(trapezoid(chord, np.cos(theta0)))  # == S divided by (b/2)
        SMC = twoS_b / 2
        b = AR * SMC

        # Setup Fourier coefficient output
        fourier_coeff = np.zeros((len(alpha_inf), N))
        for i, a_inf in enumerate(alpha_inf.flat):

            # Assume alpha = alpha_inf + alpha_geo, ignoring induced AoA for now
            alpha = a_inf + alpha_geo
            clalpha_2d = [f_clalpha_2d[j](alpha[j]) for j in range(N)]

            # LHS populate
            matA = np.zeros((len(theta0), len(theta0)))
            for j in (n := np.arange(N)):
                term1 = 4 * b / clalpha_2d[j] / chord[j]
                term2 = (n + 1) / np.sin(theta0[j])
                matA[j] += np.sin((n + 1) * theta0[j]) * (term1 + term2)

            # RHS populate
            matB = a_inf + alpha_geo - alpha_zl
            if (matB == 0).all():
                errormsg = (
                    f"Couldn't solve for PLLT's general lift distribution "
                    f"Fourier coefficients as the RHS matrix was all-zero "
                    f"(i.e. the aerofoil setup is producing zero-lift). "
                    f"This can be fixed by modifying freestream AoA, "
                    f"geometric twist in the wing, or changing some of the "
                    f"wing stations to aerofoils with different zero lift AOAs."
                )
                raise ValueError(errormsg)

            # Solve for fourier coefficients
            matX = np.linalg.solve(matA, matB).T[0]
            fourier_coeff[i] = matX

        return fourier_coeff

    def optimise_taper(
            self, C_L: float, AR: float, n_sections: int = None,
            N: int = None, constant_inner: bool = None) -> None:
        """
        An algorithm that optimises the lift distribution of the wing through
        the manipulation of chord length along the span of the wing.

        The wing stations themselves are not changed in any way, so as to best
        preserve characteristics such as prescribed stalling behaviour of the
        wing as a whole. This means while a station's chord might change, its
        given position along the span of the wing will not change.

        Args:
            C_L: The target coefficient of lift to be produced by the wing.
            AR: The aspect ratio of the final wing.
            n_sections: The number of trapezoidal sections to distribute along
                each wing of the wingplane. Optional, defaults to 1 (straight
                tapered wing).
            N: The number of sine-spaced samples to compute over the span of the
                [-1, 1] wing domain. Optional, defaults to 5-.
            constant_inner: A boolean flag, specifies whether or not the
                innermost section of the wing has a constant chord or not. This
                statement has no effect if 'n_sections' == 1 (as you would end
                up with an uninteresting rectangular wing). Optional, defaults
                to False.

        Returns:
            tuple: Non-dimensional

        """
        # Recast as necessary
        n_sections = 1 if n_sections is None else n_sections
        N = 100 if N is None else N  # ~3 s.f. precision
        constant_inner = False if constant_inner is None else constant_inner

        if n_sections < 1:
            raise ValueError(f"{n_sections=} is invalid, must be >= 1")
        elif n_sections > 3:
            warnmsg = f"n_sections > 3 (got {n_sections=}) is not advisable"
            warnings.warn(message=warnmsg, category=RuntimeWarning)

        # Initialise a solution based on elliptical planforms
        n_ctrlpts = (2 * n_sections) + 1
        my_xp = np.linspace(-1, 1, n_ctrlpts)
        my_fp = np.sin(np.arccos(my_xp))

        def factory_f_chord(xp, fp):
            """Given xp, fp, return the function numpy.interp(x, xp, fp)."""

            def procedural_f_chord(y_b2):
                """A function describing chord given position y / (b/2)."""
                return np.interp(y_b2, xp, fp)

            return procedural_f_chord

        # A function with domain [-1, 1] and range [0, 1]
        f_chord = factory_f_chord(xp=my_xp, fp=my_fp)

        # Constrain AoA optimisation solutions
        # Check that the set of elements representing AoA are of length > 1
        alpha_geo = cast2numpy([x.alpha_geo for x in self._stations.values()])
        alpha_zl = cast2numpy([x.alpha_zl for x in self._stations.values()])
        if len(set(alpha_geo - alpha_zl)) == 1:
            alpha_lo = -(alpha_geo - alpha_zl)[0] + 1e-3
            alpha_hi = float(np.radians(20))
        else:
            alpha_lo = float(-np.radians(8))
            alpha_hi = float(np.radians(20))

        # Constrain wing optimisation solutions (add bounds)
        bound_chord_lo, bound_posns_lo = np.zeros((2, n_sections + 1))
        bound_chord_hi, bound_posns_hi = np.ones((2, n_sections + 1))
        # Force wing root to be length 1, and position 0
        bound_chord_lo[0] = 1
        bound_posns_hi[0] = 0
        # Force inboard section to be constant chord
        if constant_inner is True and n_sections > 1:
            bound_chord_lo[1] = 1
        # Force wing tip position
        bound_posns_lo[-1] = 1
        # Build bounds
        bounds = sopt.Bounds(
            [x for pair in zip(bound_chord_lo, bound_posns_lo) for x in pair],
            [x for pair in zip(bound_chord_hi, bound_posns_hi) for x in pair]
        )

        # Initial solution for alpha_inf? Just to stop IDE COMPLAINING
        alpha_inf = 0.1  # ~5.7 degrees
        # This represents initial solution for x0, from port --> starboard
        x0 = np.array([x for pair in zip(my_fp, my_xp) for x in pair])
        # The wing is symmetric, so reduce solution to centreline --> starboard
        x0 = x0[2 * n_sections:]

        def parse_interspersed_x(x) -> tuple:
            """
            Given interwoven chord length and spanwise position of this chord
            value, return x parsed into respective arrays.

            Args:
                x: Interwoven chord length and spanwise position of this chord.

            Returns:
                tuple: (chord lengths, semispan-wise position of chord lengths).

            """
            # Even indices for chord length, odd indices for ctrlpt posn.
            my_chord_lengths = x[::2]  # Ordered port --> starboard
            my_ctrlpt_posns = x[1::2]  # Ordered port --> starboard

            # Since we have centreline -> starboard (symmetry), duplicate
            # and mirror arguments to produce the whole planform
            my_chord_lengths = np.concatenate(
                [my_chord_lengths[:0:-1], my_chord_lengths])
            my_ctrlpt_posns = np.concatenate(
                [-my_ctrlpt_posns[:0:-1], my_ctrlpt_posns])
            return my_chord_lengths, my_ctrlpt_posns

        steps_max = 5
        print(f"Optimising lift distribution via taper ratio(s)...", end=" ")
        print(f"({steps_max=})")
        for step in range(steps_max):

            # Step 1: optimise the freestream angle of attack that gives C_L
            def f_opt(aoa_rad) -> float:
                """Given AoA (radians), compute error w.r.t target C_L."""
                fourier_coeffs = self._compute_gld(
                    f_nd_chord=f_chord, AR=AR, alpha_inf=aoa_rad, N=N)
                matX = fourier_coeffs[0]
                cl_computed = np.pi * AR * matX[0]
                error = C_L - cl_computed

                return error

            alpha_inf = sopt.brentq(f_opt, a=alpha_lo, b=alpha_hi, xtol=1e-4)

            # Step 2: Minimise delta
            def f_opt(args: np.ndarray) -> float:
                """
                Given an array describing planform control point coordinates,
                estimate the deviation of the planform's lift distribution from
                that of the elliptical lift distribution.

                Args:
                    args: An array of the form (chord0, position0, chord1,
                        position1, ...) etc.

                Returns:
                    float: delta, from which the span efficiency factor, e, is
                        computed as follows: e = 1 / (1 + delta).

                """
                # Unpack arguments received
                my_chord_lengths, my_ctrlpt_posns = parse_interspersed_x(args)

                # Now, build the planform with linear interpolation
                my_f_chord = \
                    factory_f_chord(xp=my_ctrlpt_posns, fp=my_chord_lengths)

                # Try out the objective function - minimising delta
                fourier_coeffs = self._compute_gld(
                    f_nd_chord=my_f_chord, AR=AR, alpha_inf=alpha_inf, N=N)
                matX = fourier_coeffs[0]
                delta = (((np.arange(N) + 1) * (matX / matX[0]) ** 2)[1:]).sum()

                # Print for funsies
                e = 1 / (1 + delta)
                print(f"\rAOA={alpha_inf:.4f} [rad]; {delta=}, {e=}", end="")

                return delta

            # Solve and unpack solution as the latest guess for planform shape
            solution = sopt.minimize(
                fun=f_opt, x0=x0,  # Optimisation function and initial guess
                method="trust-constr",  # Optimisation strategy
                bounds=bounds,  # Earlier bounds
                tol=1e-5  # Objective value tolerance
            )
            chords, controlpoints = parse_interspersed_x(solution.x)
            f_chord = factory_f_chord(xp=controlpoints, fp=chords)

            # Check for convergence, and break if converged:
            # Compare current x0 to latest update to x0 (from solution)
            if np.allclose(x0, (x0 := solution.x), atol=1e-3):
                break  # 1e-3 precision * 40 m semispan = 0.04 m (4 cm) error
            # Otherwise, update problem bounds to search +/-20% of n.d. space
            else:
                new_lb = np.where(
                    bounds.lb != bounds.ub, solution.x - 0.2, bounds.lb)
                new_ub = np.where(
                    bounds.lb != bounds.ub, solution.x + 0.2, bounds.ub)
                bounds = sopt.Bounds(*np.clip([new_lb, new_ub], 0, 1))
            print("")  # Go to next line
        print(f"\n[DONE]\n")

        # Chords bound by [0, 1], control points bound by [-1, 1]
        nd_chord, nd_ctrlpt = parse_interspersed_x(x0)

        # Since S = b * Standard.Mean.Chord; S / (b/2) == 2 * (S / b) == 2 * SMC
        # And now b = AR * SMC, the span we need for a fixed aspect ratio
        twoS_b = abs(trapezoid(nd_chord, nd_ctrlpt))  # == S divided by (b/2)
        SMC = twoS_b / 2
        b = AR * SMC
        # Chords bound by [0, 1], control points bound by [-b/2, b/2]
        nd_ctrlpt = nd_ctrlpt * (b / 2)

        self._ctrlpts = nd_ctrlpt, nd_chord

        return None


class NewNDWing(object):
    """
    A class of methods for instantiating different kinds of non-dimensional wing
    geometry objects.
    """

    @classmethod
    def mainplane(cls):
        """
        Mainplanes are primarily used for lift generation.

        Returns:
            A wing object, tailored for use as a main lifting surface.

        """
        raise NotImplementedError

    @classmethod
    def stabiliser(cls):
        """
        Stabilisers are mainly used to provide pitch and yaw control authority
        in subsonic flow regimes.

        Returns:
            A wing object, makes use of moving control surfaces to demonstrate
                longitudinal (pitch) and/or directional (yaw) control.

        """
        raise NotImplementedError

    @classmethod
    def stabilator(cls):
        """
        Stabilators are mainly used to provide pitch and yaw control authority
        in vehicles where supersonic flow over a conventionally hinged control
        surface may lead to `Mach tuck'.

        Returns:
            A wing object, tailored for use as an aircraft stabilator.

        """
        raise NotImplementedError


if __name__ == "__main__":
    from carpy.aerodynamics.aerofoil import NewNDAerofoil

    aerofoil1 = NewNDAerofoil.from_url(
        "http://airfoiltools.com/airfoil/lednicerdatfile?airfoil=fx76mp140-il")
    aerofoil2 = NewNDAerofoil.from_url(
        "http://airfoiltools.com/airfoil/lednicerdatfile?airfoil=dae31-il")

    aerofoil1.alpha_zl = 0
    aerofoil2.alpha_zl = 0

    wing = NDWing()
    wing.new_station(y=10 / 12, nd_profile=aerofoil1)
    wing.new_station(y=1, nd_profile=aerofoil2)
    wing.optimise_taper(C_L=1, AR=28.1, n_sections=2, constant_inner=True)

    # import cProfile
    #
    # cProfile.run(
    #     "wing.optimise_taper(C_L=1, AR=28.1, n_sections=3, constant_inner=True)")

    nd_ctrlpt, nd_chord = wing.nd_controlpoints

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, dpi=140)

    nd_area = trapezoid(nd_chord, nd_ctrlpt)
    nd_chord = nd_chord * (20.48 / nd_area) ** 0.5
    nd_ctrlpt = nd_ctrlpt * (20.48 / nd_area) ** 0.5
    ax.plot(nd_ctrlpt, nd_chord)
    ax.set_aspect(1)
    ax.set_ylim(-1, 2)
    ax.set_xlabel("y (span position)")
    ax.set_ylabel("c (chord length)")
    ax.grid()
    ax.plot((nd_ctrlpt.min(), nd_ctrlpt.max()), (0, 0))
    plt.show()
