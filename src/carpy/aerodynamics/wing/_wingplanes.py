"""Methods for generating wing planes."""
import warnings

import numpy as np
from scipy.integrate import simpson
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

    def __mul__(self, other):
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

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot add {type(self)=} to {type(other)=}")

        new_object = type(self)(
            nd_profile=self.nd_profile + other.nd_profile,
            alpha_geo=self.alpha_geo + other.alpha_geo
        )
        return new_object

    @property
    def nd_profile(self):
        """
        The non-dimensional aerofoil object attached to this wing station.

        Returns:
            NDAerofoil object.

        """
        return self._nd_profile

    @property
    def alpha_zl(self) -> float:
        # Theoretically, documentation is inherited from the nested property???
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

    def interp_station(self, y: Hint.nums):

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
                [-1, 1] wing domain. Optional, defaults to 100.

        Returns:
            np.ndarray: An array with the same outer dimensions as alpha_inf,
                and deepest inner dimension is the Fourier series solution.

        """

        # Recast as necessary
        alpha_inf = np.zeros(1) if alpha_inf is None else cast2numpy(alpha_inf)
        N = 100 if N is None else N  # ~3 d.p. precision

        theta0 = np.linspace(0, np.pi, N + 2)[1:-1]  # Arguments of distribution

        # Setup some arrays for computation
        interp_stations = self.interp_station(y=np.abs(np.cos(theta0)))
        chord = f_nd_chord(np.cos(theta0))
        alpha_geo = cast2numpy([[x.alpha_geo] for x in interp_stations])
        alpha_zl = cast2numpy([[0] for _ in theta0], dtype=np.float64)
        f_clalpha_2d = [lambda x: 2 * np.pi for _ in theta0]
        warnmsg = (
            f"Assuming zero-lift angle of attack of zero degrees for all "
            f"aerofoils (all aerofoils are considered thin, flat plates)"
        )
        warnings.warn(message=warnmsg, category=RuntimeWarning)
        warnmsg = (
            f"Assuming lift curve slope of CLa = 2 * pi at all AOA, for all "
            f"aerofoil sections (as per 2D flat plate theory)"
        )
        warnings.warn(message=warnmsg, category=RuntimeWarning)

        # Since S = b * Standard.Mean.Chord; S / (b/2) == 2 * (S / b) == 2 * SMC
        # And now b = AR * SMC, the span we need for a fixed aspect ratio
        twoS_b = abs(simpson(chord, np.cos(theta0)))  # == S divided by (b/2)
        SMC = twoS_b / 2
        b = AR * SMC

        # Setup Fourier coefficient output
        fourier_coeff = np.zeros((len(alpha_inf), N))
        for i, a_inf in enumerate(alpha_inf.flat):

            # Assume alpha = alpha_inf + alpha_geo, ignoring induced AoA for now
            alpha = a_inf + alpha_geo
            clalpha_2d = [f_clalpha_2d[j](alpha) for j in range(N)]

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

        # # Compute ELD deviation, delta
        # matX = fourier_coeff[0]
        # delta = (((np.arange(N) + 1) * (matX / matX[0]) ** 2)[1:]).sum()
        # e = 1 / (1 + delta)
        # print(f"{delta=}, {e=}")

        return fourier_coeff

    def optimise(
            self, C_L: float, AR: float, n_sections: int = None,
            N: int = None) -> tuple[np.ndarray, np.ndarray]:
        """
        A gradient descent algorithm that optimises the lift distribution of the
        wing by the manipulation of chord length along the span of the wing.

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
                [-1, 1] wing domain. Optional, defaults to 100.

        Returns:
            tuple: Non-dimensional

        """
        # Recast as necessary
        n_sections = 1 if n_sections is None else n_sections
        N = 100 if N is None else N  # ~3 d.p. precision

        # Initialise a solution based on elliptical planforms
        n_ctrlpts = (2 * n_sections) + 1
        xp = np.linspace(-1, 1, n_ctrlpts)
        fp = np.sin(np.arccos(xp))
        f_chord = lambda x: np.interp(x, xp, fp)

        # Constrain AoA optimisation solutions
        # Check that the set of elements representing AoA are of length > 1
        alpha_geo = cast2numpy([x.alpha_geo for x in self._stations.values()])
        alpha_zl = cast2numpy([0 for _ in alpha_geo], dtype=np.float64)
        if len(set(alpha_geo - alpha_zl)) == 1:
            alpha_lo = -(alpha_geo - alpha_zl)[0] + 1e-3
            alpha_hi = float(np.radians(20))
        else:
            alpha_lo = float(-np.radians(8))
            alpha_hi = float(np.radians(20))
        warnmsg = (
            f"Assuming zero-lift angle of attack of zero degrees for all "
            f"aerofoils (all aerofoils are considered thin, flat plates)"
        )
        warnings.warn(message=warnmsg, category=RuntimeWarning)

        # Constrain wing optimisation solutions
        # package boundaries as (chord0, position0, chord1, position1, ...)
        bound_chord_lo = np.zeros(n_ctrlpts)
        bound_chord_hi = np.ones(n_ctrlpts)
        bound_chord_lo[n_sections] = 1  # Make root chord unchangeable length
        bound_posn_lo = -np.ones(n_ctrlpts)
        bound_posn_hi = np.ones(n_ctrlpts)
        bound_posn_hi[:n_sections] = 0  # Anything port is bounded by centreline
        bound_posn_lo[n_sections:] = 0  # Anything stbd is bounded by centreline
        bound_posn_hi[n_sections:] = 1  # Anything stbd is bounded by wingtip
        bound_posn_lo[n_sections] = 0  # Fix the root control point position
        bound_posn_hi[n_sections] = 0
        bound_posn_lo[-1] = 1  # Fix the starboard tip control point position
        bound_posn_hi[-1] = 1
        bound_posn_lo[0] = -1  # Fix the port tip control point position
        bound_posn_hi[0] = -1
        # Bounds currently represent full span. Because of mirror symmetry, we
        # can reduce the size of the problem to just centreline --> starboard
        bound_chord_lo = bound_chord_lo[n_sections:]
        bound_chord_hi = bound_chord_hi[n_sections:]
        bound_posn_lo = bound_posn_lo[n_sections:]
        bound_posn_hi = bound_posn_hi[n_sections:]
        bounds = sopt.Bounds(
            [x for pair in zip(bound_chord_lo, bound_posn_lo) for x in pair],
            [x for pair in zip(bound_chord_hi, bound_posn_hi) for x in pair]
        )
        # This represents x0 for port --> starboard
        x0 = np.array([x for pair in zip(fp, xp) for x in pair])
        # The wing is symmetric, so reduce to centreline --> starboard
        x0 = x0[2 * n_sections:]

        steps_max = 5
        for step in range(steps_max):

            print(f"\rOptimisation step {step + 1} of {steps_max=}...", end="")

            # Step 1: optimise the freestream angle of attack that gives C_L
            def f_opt(aoa_rad):
                fourier_coeffs = self._compute_gld(
                    f_nd_chord=f_chord, AR=AR, alpha_inf=aoa_rad, N=N)
                matX = fourier_coeffs[0]
                cl_computed = np.pi * AR * matX[0]
                error = C_L - cl_computed
                return error

            alpha_inf = sopt.brentq(f_opt, a=alpha_lo, b=alpha_hi, xtol=1e-3)

            # Step 2: Minimise delta
            def f_opt(args):
                # Unpack arguments received
                # Even indices for chord length, odd indices for ctrlpt posn.
                chord_lengths = args[::2]  # Ordered port --> starboard
                ctrlpt_posns = args[1::2]  # Ordered port --> starboard

                # Since we have centreline -> starboard (symmetry), duplicate
                # and mirror arguments to produce the whole planform
                chord_lengths = np.concatenate(
                    [chord_lengths[:0:-1], chord_lengths])
                ctrlpt_posns = np.concatenate(
                    [-ctrlpt_posns[:0:-1], ctrlpt_posns])

                # Now, build the planform with linear interpolation
                f_chord = lambda x: np.interp(x, ctrlpt_posns, chord_lengths)

                # Try out the objective function - minimising delta
                fourier_coeffs = self._compute_gld(
                    f_nd_chord=f_chord, AR=AR, alpha_inf=alpha_inf, N=N)
                matX = fourier_coeffs[0]
                delta = (((np.arange(N) + 1) * (matX / matX[0]) ** 2)[1:]).sum()

                return delta

            # Solve, then check for convergence
            solution = sopt.minimize(
                fun=f_opt, x0=x0,  # Optimisation function and initial guess
                method="trust-constr",  # Optimisation strategy
                bounds=bounds  # Earlier bounds
            )
            # In break and continue cases, update solution *after* comparison
            if np.allclose(x0, solution.x):
                x0 = solution.x
                break
            else:
                x0 = solution.x
                continue
        print(f"planform optimisation complete.")

        chord_lengths = x0[::2]  # Ordered port --> starboard
        ctrlpt_posns = x0[1::2]  # Ordered port --> starboard

        # Since we have centreline -> starboard (symmetry), duplicate
        # and mirror arguments to produce the whole planform
        chord_lengths = np.concatenate(
            [chord_lengths[:0:-1], chord_lengths])
        ctrlpt_posns = np.concatenate(
            [-ctrlpt_posns[:0:-1], ctrlpt_posns])

        # Perhaps because I already have aspect ratio, I should just return
        # a 2D array of coordinates per metre span, so a user can then scale as
        # necessary to get the right wing area...

        return ctrlpt_posns, chord_lengths


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

    aerofoil1 = NewNDAerofoil.from_procedure.NACA(code="0012")
    aerofoil2 = NewNDAerofoil.from_procedure.NACA(code="0012")

    wing = NDWing()
    wing.new_station(y=0.9, nd_profile=aerofoil1)
    wing.new_station(y=0.2, nd_profile=aerofoil2, alpha_geo=np.radians(0))
    wing.optimise(C_L=1.1, AR=28, n_sections=2)
