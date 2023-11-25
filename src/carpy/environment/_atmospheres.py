"""Module implementing models of the Earth's atmosphere."""
import functools
import itertools
import os
import re
import warnings

import numpy as np
import pandas as pd

from carpy.gaskinetics import GasModels, IsentropicFlow
from carpy.utility import (
    GetPath, Hint, Pretty, Quantity,
    cast2numpy, cast2quantity, constants as co, isNone, interp_lin, interp_exp
)

__all__ = [
    "LIBREF_ATM", "ISA1975", "US1976", "MILHDBK310", "ObsAtmospherePerfect"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Load data required by the module
# ---------------------------------------------------------------------------- #
def load_iso2533(filename: str, /):
    """Load and return dataframes for the ISO 2533:1975 standard."""
    filepath = os.path.join(GetPath.localpackage(), "data", filename)
    dataframes = pd.read_excel(io=filepath, sheet_name=None, na_filter=False)
    return dataframes


df_ISA1975 = load_iso2533("ISO-2533-1975.xlsx")


def load_milhdbk310(filename: str, /):
    """Load and return dataframes for the MIL-HDBK-310 specification."""
    filepath = os.path.join(GetPath.localpackage(), "data", filename)

    def read_as_number(x):
        """Read and parse numbers as they are given in the MIL-HDBK-310 spec."""
        x = str(x)
        components = re.findall(r"\d+\.?(?:\d+)?|[+-]\d", x)
        if len(components) == 1:
            return float(components[0])
        else:
            return eval(f"{components[0]}e{components[1]}")

    columns = pd.read_excel(io=filepath, nrows=0).columns
    dataframes = pd.read_excel(io=filepath, sheet_name=None, na_filter=False,
                               converters={k: read_as_number for k in columns})
    return dataframes


df_MILHDBK310 = load_milhdbk310("MIL-HDBK-310_profiles.xlsx")


# ============================================================================ #
# Support functions
# ---------------------------------------------------------------------------- #

def atmosphere_builder(altitude: Hint.nums, T: Hint.nums = None,
                       p: Hint.nums = None, rho: Hint.nums = None,
                       Mbar: float = None):
    """
    Return functions for temperature, pressure, and density with altitude.

    Given data for altitude and any valid combinations of temperature, pressure,
    and density at the associated altitude levels, return functions that accept
    altitude as an argument and return the corresponding values for T, p, and
    rho.

    Args:
        altitude: Altitude data. Assumed to be increasing.
        T: Temperature data, in the same shape as the altitude data. Optional.
        p: Pressure data, in the same shape as the altitude data. Optional.
        rho: Density data, in the same shape as the altitude data. Optional.
        Mbar: Average molar mass of atmospheric particles. Optional.

    Returns:
        f_temperature(altitude), f_pressure(altitude), f_density(altitude).

    """
    # Recast as necessary
    altitude = Quantity(altitude, "m")
    T = None if T is None else Quantity(T, "K")
    p = None if p is None else Quantity(p, "Pa")
    rho = None if rho is None else Quantity(rho, "kg m^{-3}")
    Mbar = co.STANDARD.SL.M if Mbar is None else Quantity(Mbar, "kg mol^{-1}")
    # pre-package kwargs for interpolation and extrapolation of atm. quantities
    commonkwargs = dict([("xp", altitude), ("bounded", True)])

    # Verify that altitude data is increasing
    if all(np.diff(altitude) <= 0):
        errormsg = f"Elements of 'altitude' data must be increasing"
        raise ValueError(errormsg)

    # If temperature is provided and no other arguments are present
    if T is not None and all(isNone(p, rho)):

        # Because pressure is not given, we must assume that an altitude of zero
        # corresponds with standard sea-level pressure. We have to make sure
        # zero appears in the altitude array (with corresponding temperature).
        idx_of_SL = len(altitude[altitude < 0])
        if 0 not in altitude:
            # Make sure 0 appears in the altitude listing and recast temperature
            altitude_with_zero = np.insert(altitude, idx_of_SL, 0)
            T = interp_lin(altitude_with_zero, altitude, T)
            altitude = altitude_with_zero

        # Initialise base-layer pressure
        p0 = co.STANDARD.SL.p
        T_lapserate = np.diff(T) / np.diff(altitude)
        for i in range(idx_of_SL):
            ui = idx_of_SL - i
            li = idx_of_SL - (i + 1)
            daltitude = altitude[ui] - altitude[li]

            # Molecular-scale pressure equation depends on lapse rate
            if T_lapserate[i] != 0:
                exponent = np.asarray(
                    co.STANDARD.SL.g / co.STANDARD.SL.Rs / T_lapserate[li])
                p0 /= (T[li] / (T[ui])) ** exponent
            else:
                exponent = np.asarray(
                    -co.STANDARD.SL.g * daltitude / co.STANDARD.SL.Rs / T[li])
                p0 /= np.exp(exponent)

        # Define functions
        f_T = functools.partial(interp_lin, fp=T, **commonkwargs)

        def f_p(y):
            """Atmospheric pressure as a function of altitude."""
            # Initialise values to lowest altitude
            Ts = np.ones(y.shape) * T[0]
            ps = np.ones(y.shape) * p0

            for j in range(len(altitude) - 1):
                # Clip alt. by the layer's upper bound, and diff. to lower bound
                # If you're a future developer looking to find out how to
                # extrapolate atmospheric pressure modelling, this is what's
                # limiting the atmospheric model.
                dy = np.clip(
                    np.clip(y, None, altitude[j + 1]) - altitude[j],
                    0, None
                )
                # Molecular-scale pressure equation depends on lapse rate
                if T_lapserate[j] != 0:
                    ps *= (Ts / (Ts + T_lapserate[j] * dy)) ** np.asarray(
                        co.STANDARD.SL.g / co.STANDARD.SL.Rs / T_lapserate[j])
                else:
                    ps *= np.exp(np.asarray(
                        -co.STANDARD.SL.g * dy / co.STANDARD.SL.Rs / Ts))

                # Update temperature after the fact, as we needed temperature
                # from the base of the layer to make the computation
                Ts += T_lapserate[j] * dy

            return ps

        def f_rho(y):
            """Compute density, given altitude"""
            return f_p(y) * Mbar / co.PHYSICAL.R / f_T(y)

    elif not all(isNone(T, p)) and rho is None:
        f_T = functools.partial(interp_lin, fp=T, **commonkwargs)
        f_p = functools.partial(interp_exp, fp=p, **commonkwargs)

        def f_rho(y):
            """Compute density, given altitude"""
            return f_p(y) * Mbar / co.PHYSICAL.R / f_T(y)

    elif not all(isNone(T, rho)) and p is None:
        f_T = functools.partial(interp_lin, fp=T, **commonkwargs)
        f_rho = functools.partial(interp_exp, fp=rho, **commonkwargs)

        def f_p(y):
            """Compute pressure, given altitude"""
            return f_rho(y) / (Mbar / co.PHYSICAL.R / f_T(y))

    else:
        errormsg = "Got insufficient data to generate p, T, and rho functions"
        raise ValueError(errormsg)

    return f_T, f_p, f_rho


def cast_altitudes(f_geometric: bool):
    """
    Create and return a decorator that permits use of geometric or gravitational
    potential altitudes with atmospheric property models.

    Args:
        f_geometric: Boolean flag to indicate whether the function being
            decorated is defined in terms of geometric or potential altitude.
    """

    def decorator(method):
        """Create, apply, and return the method's wrapper."""

        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            """Augment atmospheric modelling functions with default behaviour"""
            # Find all arguments
            all_args = args + tuple(v for (k, v) in kwargs.items())

            # Make sure positional argument is given
            instance, altitude = all_args[0:2]
            if altitude is None:
                errormsg = (
                    f"{type(instance).__name__}.{method.__name__}() missing 1 "
                    f"required positional argument: 'altitude'"
                )
                raise TypeError(errormsg)

            # Recast as necessary
            altitude = cast2numpy(altitude)
            if len(args) < 3:
                geometric = kwargs.get("geometric", None)
                geometric = False if geometric is None else geometric
            elif len(args) == 3:
                geometric = args[2]
            else:
                errormsg = (
                    f"{type(instance).__name__}.{method.__name__}() takes 2 "
                    f"positional arguments but {len(args) - 1} were given"
                )
                raise TypeError(errormsg)

            # Function uses geometric inputs, and user specified geometric input
            if f_geometric and geometric:
                return method(altitude)
            # Function uses geometric inputs, and user specified potential input
            elif f_geometric and not geometric:
                return method(instance.geop2geom(altitude))
            # Function uses potential inputs, and user specified geometric input
            elif not f_geometric and geometric:
                return method(instance.geom2geop(altitude))
            # Function uses potential inputs, and user specified potential input
            else:
                return method(altitude)

        return wrapper

    return decorator


# ============================================================================ #
# Atmosphere objects
# ---------------------------------------------------------------------------- #
# Base objects

class Atmosphere(object):
    """Class for models of the Earth's atmosphere based on perfect gases."""

    _f_T = NotImplemented
    _f_p = NotImplemented
    _f_rho = NotImplemented
    _f_c_sound = NotImplemented
    _f_mu_visc = NotImplemented
    _f_k_thermal = NotImplemented
    _f_gamma = NotImplemented
    _r_Earth = co.STANDARD.ISO_2533.r  # Default nominal radius of Earth

    def __init__(self, T_offset: float = None):
        self._T_offset = float(0 if T_offset is None else T_offset)
        return

    def __repr__(self):
        reprstring = f"{type(self).__name__}(Toffset={self.T_offset})"
        return reprstring

    def __str__(self):
        rtnstring = (
            f"{type(self).__name__}({Pretty.temperature(self.T_offset)})")
        return rtnstring

    @functools.cached_property
    def T_offset(self) -> Quantity:
        """
        Temperature offset compared to the original atmosphere model. A T_offset
        of -10 degrees Celsius (-10 Kelvin) indicates that the atmosphere is 10
        degrees cooler than usual.

        Returns:
            Atmospheric temperature offset.

        """
        return Quantity(self._T_offset, "K")

    @classmethod
    def geom2geop(cls, altitude: Hint.nums) -> Quantity:
        """
        Compute gravitational potential altitude given the geometric altitude.

        Args:
            altitude: Geometric altitude(s) to be converted.

        Returns:
            Geopotential altitude(s).

        """
        # Recast as necessary
        geom = Quantity(altitude, "m")

        # Compute geopotential altitude
        geop = cls._r_Earth * geom / (cls._r_Earth + geom)

        return geop

    @classmethod
    def geop2geom(cls, altitude: Hint.nums) -> Quantity:
        """
        Compute geometric altitude given the gravitational potential altitude.

        Args:
            altitude: Gravitational potential altitude(s) to be converted.

        Returns:
            Geometric altitude(s).

        """
        # Recast as necessary
        geop = Quantity(altitude, "m")

        # Compute geometric altitude
        geom = cls._r_Earth * geop / (cls._r_Earth - geop)

        return geom

    @classmethod
    def g_acc(cls, altitude: Hint.nums, phi: Hint.nums = None) -> Quantity:
        """
        Return the acceleration due to gravity.

        Args:
            altitude: The geometric altitude above mean sea level.
            phi: Latitude of the query. Optional, defaults to 45 32' 33" degrees
                North.

        Returns:
            Acceleration due to gravity.

        """
        # Recast as necessary
        altitude = cast2numpy(altitude)
        # 45 32'33" North
        phi = cast2numpy(np.radians(45.5425) if phi is None else phi)

        # Lambert's equation for acceleration of free-fall with latitude
        cosphi2 = np.cos(2 * phi)
        g_phi = Quantity(
            9.806_16 * (1 - 0.002_637_3 * cosphi2 + 0.000_005_9 * cosphi2 ** 2),
            "m s^{-2}"
        )

        # Lapse of acceleration with altitude
        g = g_phi * cls._r_Earth ** 2 / (cls._r_Earth + altitude) ** 2

        return g

    def T(self, altitude: Hint.nums, geometric: bool = False) -> Quantity:
        """
        Return the outside air temperature with altitude.

        Args:
            altitude: Altitude at which to query the atmosphere model. By
                default, the altitude is considered geopotentially scaled
                (unless otherwise specified using the 'geometric' argument).
            geometric: Flag specifying if given altitudes are geometric or not.
                Optional, defaults to False.

        Returns:
            Outside air temperature.

        """
        offset = self.T_offset
        return Quantity(self._f_T(self, altitude, geometric) + offset, "K")

    def p(self, altitude: Hint.nums, geometric: bool = False) -> Quantity:
        """
        Return the outside static air pressure with altitude.

        Args:
            altitude: Altitude at which to query the atmosphere model. By
                default, the altitude is considered geopotentially scaled
                (unless otherwise specified using the 'geometric' argument).
            geometric: Flag specifying if given altitudes are geometric or not.
                Optional, defaults to False.

        Returns:
            Outside static air pressure.

        """
        return Quantity(self._f_p(self, altitude, geometric), "Pa")

    def rho(self, altitude: Hint.nums, geometric: bool = False) -> Quantity:
        """
        Return the outside air density with altitude.

        Args:
            altitude: Altitude at which to query the atmosphere model. By
                default, the altitude is considered geopotentially scaled
                (unless otherwise specified using the 'geometric' argument).
            geometric: Flag specifying if given altitudes are geometric or not.
                Optional, defaults to False.

        Returns:
            Outside air density.

        """
        _ = geometric  # A "do nothing" statement, geometric is for the wrapper
        return Quantity(self._f_rho(self, altitude, geometric), "kg m^{-3}")

    def gamma(self, altitude: Hint.nums, geometric: bool = False) -> Quantity:
        """
        Return the ratio of specific heats of air with altitude.

        Args:
            altitude: Altitude at which to query the atmosphere model. By
                default, the altitude is considered geopotentially scaled
                (unless otherwise specified using the 'geometric' argument).
            geometric: Flag specifying if given altitudes are geometric or not.
                Optional, defaults to False.

        Returns:
            Adiabatic index of air.

        """
        return self._f_gamma(altitude, geometric)

    def c_sound(self, altitude: Hint.nums, geometric: bool = False) -> Quantity:
        """
        Return the speed of sound in air with altitude.

        Args:
            altitude: Altitude at which to query the atmosphere model. By
                default, the altitude is considered geopotentially scaled
                (unless otherwise specified using the 'geometric' argument).
            geometric: Flag specifying if given altitudes are geometric or not.
                Optional, defaults to False.

        Returns:
            Speed of sound in air.

        """
        return Quantity(self._f_c_sound(altitude, geometric), "m s^{-1}")

    def mu_visc(self, altitude: Hint.nums, geometric: bool = False) -> Quantity:
        """
        Return the dynamic viscosity of air with altitude.

        Args:
            altitude: Altitude at which to query the atmosphere model. By
                default, the altitude is considered geopotentially scaled
                (unless otherwise specified using the 'geometric' argument).
            geometric: Flag specifying if given altitudes are geometric or not.
                Optional, defaults to False.

        Returns:
            Dynamic viscosity of air.

        """
        return Quantity(self._f_mu_visc(altitude, geometric), "Pa s")

    def _f_nu(self, altitude: Hint.nums, geometric: bool = False) -> Quantity:
        """Return the kinematic viscosity of air with altitude."""
        # Take advantage of the relationship with dynamic viscosity
        kwargs = {"altitude": altitude, "geometric": geometric}
        return self._f_mu_visc(**kwargs) / self._f_rho(**kwargs)

    def nu_visc(self, altitude: Hint.nums, geometric: bool = False) -> Quantity:
        """
        Return the kinematic viscosity of air with altitude.

        Args:
            altitude: Altitude at which to query the atmosphere model. By
                default, the altitude is considered geopotentially scaled
                (unless otherwise specified using the 'geometric' argument).
            geometric: Flag specifying if given altitudes are geometric or not.
                Optional, defaults to False.

        Returns:
            Kinematic viscosity of air.

        """
        # Take advantage of the relationship with dynamic viscosity
        kwargs = {"altitude": altitude, "geometric": geometric}
        return self._f_mu_visc(**kwargs) / self._f_rho(**kwargs)

    def k_thermal(self, altitude: Hint.nums,
                  geometric: bool = False) -> Quantity:
        """
        Return the thermal conductivity of air with altitude.

        Args:
            altitude: Altitude at which to query the atmosphere model. By
                default, the altitude is considered geopotentially scaled
                (unless otherwise specified using the 'geometric' argument).
            geometric: Flag specifying if given altitudes are geometric or not.
                Optional, defaults to False.

        Returns:
            Thermal conductivity of air.

        """
        return self._f_k_thermal(altitude, geometric)

    def airspeeds(self, altitude: Hint.nums = None, geometric=False, *,
                  CAS: Hint.nums = None, EAS: Hint.nums = None,
                  TAS: Hint.nums = None, Mach: Hint.nums = None) -> tuple:
        """
        Return calibrated, equivalent, and true airspeed at the query altitude.

        Args:
            altitude: Altitude at which to query the atmosphere model. By
                default, the altitude is considered geopotentially scaled
                (unless otherwise specified using the 'geometric' argument).
            geometric: Flag specifying if given altitudes are geometric or not.
                Optional, defaults to False.
            CAS: Calibrated (indicated) airspeed.
            EAS: Equivalent airspeed at sea level.
            TAS: True airspeed of the fluid.
            Mach: The Mach number of flight.

        Returns:
            tuple: (CAS, EAS, TAS, M) in metres per second (where applicable).

        """
        # Recast inputs as necessary
        altitude = cast2numpy(altitude)
        kwargs = {"altitude": altitude, "geometric": geometric}
        CAS = CAS if CAS is None else Quantity(CAS, "m s^{-1}")
        EAS = EAS if EAS is None else Quantity(EAS, "m s^{-1}")
        TAS = TAS if TAS is None else Quantity(TAS, "m s^{-1}")
        Mach = Mach if Mach is None else cast2numpy(Mach)

        # Verify validity of inputs, and that only one input is given
        if tuple(isNone(CAS, EAS, TAS, Mach)).count(True) == 3:
            pass
        else:
            errormsg = (
                f"Expected one of CAS, EAS, TAS, and Mach arguments should be "
                f"used (got {CAS=}, {EAS=}, {TAS=}, {Mach=})"
            )
            raise ValueError(errormsg)

        # Solve for the missing speeds!
        rho = self.rho(**kwargs)
        rho_sl = self.rho(altitude=0)
        p = self.p(**kwargs)
        p_sl = self.p(altitude=0)
        a = self.c_sound(**kwargs)
        a_sl = self.c_sound(altitude=0)
        gamma = self.gamma(**kwargs)
        while True:
            # EAS can be computed from TAS
            if EAS is None and TAS is not None:
                EAS = TAS * np.sqrt(rho / rho_sl)
            # TAS can be computed from EAS
            elif TAS is None and EAS is not None:
                TAS = EAS / np.sqrt(rho / rho_sl)
            # CAS can be computed from TAS
            elif CAS is None and EAS is not None and TAS is not None:
                # Use isentropic flow relations to find the Mach number
                Mach = TAS / a
                pt_p = 1 / IsentropicFlow.p_pt(M=Mach, gamma=gamma)
                qc = p * (pt_p - 1)
                Tt_Tsl = (qc / p_sl + 1) ** ((gamma - 1) / gamma)
                Tt_T = (qc / p + 1) ** ((gamma - 1) / gamma)
                # If Mach is zero, there is a divide by zero error. Suppress it.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    CAS = EAS * ((p_sl / p) * (Tt_Tsl - 1) / (Tt_T - 1)) ** 0.5
                CAS = np.where(np.isnan(CAS), 0, CAS)  # replace the nans with 0
            # TAS can be computed from CAS
            elif TAS is None and CAS is not None:
                # Use isentropic flow relations to find the Mach number
                M_sl = CAS / a_sl
                pt_psl = 1 / IsentropicFlow.p_pt(M=M_sl, gamma=gamma)
                qc = p_sl * (pt_psl - 1)
                Mach = IsentropicFlow.M(p_p0=(p / (p + qc)), gamma=gamma)
                TAS = Mach * a
            # TAS can be computed from Mach
            elif TAS is None and Mach is not None:
                TAS = Mach * a
            # It is only possible to reach this stage if we have all the params
            else:
                break  # Leave the while loop

        # Find the shape of the output array by finding the biggest shape
        oshape = max([
            x.shape for x in [altitude, CAS, EAS, TAS, Mach] if x is not None])
        # These speed arrays are flattened, 1D and need to be cast back to shape
        speeds = np.array(np.broadcast_arrays(CAS, EAS, TAS, Mach))

        # Cast back into the right shape
        speeds = tuple([
            (Quantity(x, "m s^{-1}") if i != 3 else x).reshape(oshape)
            for i, x in enumerate(speeds)
        ])
        return speeds


# ---------------------------------------------------------------------------- #
# International Standard Atmosphere (1975)

class ISA1975(Atmosphere):
    """
    Simple model of the International Standard Atmosphere, as given in the
    ISO 2533:1975 standard.
    """

    def __init__(self, T_offset: float = None):
        # Super call
        super().__init__(T_offset=T_offset)

        # Get fundamental atmopsheric properties (in terms of geopotential alt.)
        table4 = cast2quantity(df_ISA1975["Table4"])
        f_temperature, f_pressure, f_density = atmosphere_builder(
            altitude=table4["Geopotential altitude, H"],
            T=table4["Temperature, T"],
            Mbar=co.STANDARD.ISO_2533.M
        )
        self._f_T = cast_altitudes(f_geometric=False)(f_temperature)
        self._f_p = cast_altitudes(f_geometric=False)(f_pressure)
        self._f_rho = cast_altitudes(f_geometric=False)(f_density)

        # Estimate the specific gas constant, gamma
        X = (("N2", 78.084), ("O2", 20.947_6), ("Ar", 0.934), ("CO2", 0.0314))
        X = ", ".join([f"{species}:{amount}" for (species, amount) in X])
        gas = GasModels.PerfectCaloric()
        gas.X = X
        self._f_gamma = lambda *args, **kwargs: gas.gamma

        return

    def _f_c_sound(self, altitude: Hint.nums,
                   geometric: bool = False) -> Quantity:
        """Return the speed of sound in air with altitude."""
        T = self.T(altitude, geometric)
        gamma = self.gamma(altitude, geometric)
        a = (gamma * co.STANDARD.ISO_2533.Rs * T) ** 0.5
        return a

    def _f_mu_visc(self, altitude: Hint.nums,
                   geometric: bool = False) -> Quantity:
        """Return the kinematic viscosity of air with altitude."""
        # Use Sutherland's law of viscosity
        T = self.T(altitude, geometric)
        mu = co.STANDARD.ISO_2533.betaS * T ** 1.5
        mu = mu / (T + co.STANDARD.ISO_2533.S)
        return mu


# Instantiate a single ISA1975, reference library atmosphere
LIBREF_ATM = ISA1975(T_offset=0)


# ---------------------------------------------------------------------------- #


# U.S. Standard Atmosphere (1976)

class US1976(Atmosphere):
    """
    Simple model of the U.S. Standard Atmosphere (1976), as given in the
    NASA-TM-X-74335 specification.
    """

    def __init__(self, T_offset: float = None):
        # Super call
        super().__init__(T_offset=T_offset)

        raise NotImplementedError("Sorry! Coming soon\u2122")


# ---------------------------------------------------------------------------- #
# U.S. Department of Defense MIL-HDBK-310

milhdbk310_atms = dict()
for sheet, x_pct in itertools.product(list(df_MILHDBK310), ["1pct", "10pct"]):
    # Get fundamental atmopsheric properties (in terms of geometric alt.)
    table = cast2quantity(df_MILHDBK310[sheet])
    f_mil310_T, f_mil310_p, f_mil310_rho = atmosphere_builder(
        altitude=table["Geometric altitude, Z"],
        T=table[f"Temperature {x_pct}, T"],
        rho=table[f"Density {x_pct}, rho"],
        Mbar=co.STANDARD.ISO_2533.M
    )
    to_set_inside_init = dict([
        ("_f_T", cast_altitudes(f_geometric=True)(f_mil310_T)),
        ("_f_p", cast_altitudes(f_geometric=True)(f_mil310_p)),
        ("_f_rho", cast_altitudes(f_geometric=True)(f_mil310_rho)),
        ("_f_gamma", lambda *args, **kwargs: 1.4),  # Assume perfect gamma=1.4
        ("_f_c_sound", LIBREF_ATM.c_sound),  # Assume same method as ISA1975
        ("_f_mu_visc", LIBREF_ATM.mu_visc),  # Assume same method as ISA1975
        ("_f_k_thermal", LIBREF_ATM.k_thermal)  # Assume same method as ISA1975
    ])


    def _init(self, T_offset: float = None):
        """Dynamic initialisation function for MIL-HDBK-310 atmospheres."""
        # Super call
        Atmosphere.__init__(self, T_offset=T_offset)
        # We must set attributes like this so that the wrapper functions
        # properly (it needs to take self as an argument, be called inside init)
        for _, (k, v) in enumerate(to_set_inside_init.items()):
            setattr(self, k, v)
        return


    def _repr(self):
        """__repr__ string in the style of 'MILHDBK310_HD10_10pct'."""
        reprstring = f"{type(self).__name__}"
        return reprstring


    def _str(self):
        """__str__ string in the style of 'MILHDBK310[HD10:10%]'."""
        components = f"{type(self).__name__}".split("_")
        return f"{components[0]}[{components[1]}:{components[2][:-3]}%]"


    # Dynamically produce a class that permits access to the above properties
    attrs = dict([("__init__", _init), ("__repr__", _repr), ("__str__", _str)])
    model = type(f"MILHDBK310_{sheet}_{x_pct}", (Atmosphere,), attrs)
    milhdbk310_atms[f"{sheet}_{x_pct}"] = model(T_offset=0)  # Instantiate it!


# Doesn't matter what the name of the parent is, define a catalogue of atms.
class MILHDBK310(type("catalogue", (object,), milhdbk310_atms)):
    """A collection of atmospheric property models as given in MIL-HDBK-310."""

    def __init__(self):
        errormsg = (
            "This is a catalogue of atmospheres, and it should not be "
            "instantiated directly. Try one of my attributes!"
        )
        raise RuntimeError(errormsg)


# ---------------------------------------------------------------------------- #
# User "observed" atmospheres

class ObsAtmospherePerfect(Atmosphere):
    """
    Simple model for creating user-defined atmospheres based on observed
    atmospheric profiles (where the atmospheric gas components are assumed to
    behave ideally).
    """

    def __init__(self, altitude: Hint.nums = None, T: Hint.nums = None,
                 p: Hint.nums = None, rho: Hint.nums = None,
                 geometric: bool = None, Mbar: float = None,
                 T_offset: float = None):
        """
        Args:
            altitude: Sequence of altitudes.
            T: Sequence of temperatures. Optional, at least one of T,p, or rho
                is required.
            p: Sequence of pressures. Optional, at least one of T,p, or rho
                is required.
            rho: Sequence of densities. Optional, at least one of T,p, or rho
                is required.
            geometric: Whether the altitude sequence is geopotential or
                geometric in nature. Optional, defaults to False (geopotential).
            Mbar: The average molar mass of the atmosphere, per mol. Optional,
                defaults to that of the standard atmosphere.
            T_offset: Temperature profile offset from that in the temperature
                sequence. Optional, defaults to 0 offset.
        """
        # Super call
        super().__init__(T_offset=T_offset)

        # Recast as necessary
        geometric = False if geometric is None else geometric

        # Create functions of atmospheric properties with altitude
        f_temperature, f_pressure, f_density = atmosphere_builder(
            altitude=altitude, T=T, p=p, rho=rho, Mbar=Mbar)
        self._f_T = cast_altitudes(f_geometric=geometric)(f_temperature)
        self._f_p = cast_altitudes(f_geometric=geometric)(f_pressure)
        self._f_rho = cast_altitudes(f_geometric=geometric)(f_density)

        return
