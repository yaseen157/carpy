"""Module implementing Mattingly's basic models for thrust correction."""
import warnings

import numpy as np

from carpy.environment import LIBREF_ATM
from carpy.gaskinetics import IsentropicFlow as IFlow
from carpy.utility import constants as co, Hint, Quantity, cast2numpy

__all__ = ["MattinglyBasic"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Support functions
# ---------------------------------------------------------------------------- #

def TPratio(altitude, geometric, atmosphere):
    """Compute and return standard-day temperature and pressure ratios."""

    # Actual (static) conditions
    T = atmosphere.T(altitude=altitude, geometric=geometric)
    p = atmosphere.p(altitude=altitude, geometric=geometric)

    theta = (T / co.STANDARD.SL.T).x
    delta = (p / co.STANDARD.SL.p).x

    return theta, delta


# ============================================================================ #
# Engine Deck objects
# ---------------------------------------------------------------------------- #
# Base objects

class EngineDeck(object):
    """Class for models of engines or "performance decks"."""

    _f_Tlapse = NotImplemented
    _f_TSFC = NotImplemented

    @staticmethod
    def _parse_arguments(Mach, altitude, geometric, atmosphere):
        # Recast as necessary
        Mach = cast2numpy(Mach, dtype=np.float64)  # dtype=float permits slicing
        altitude = cast2numpy(0.0 if altitude is None else altitude)
        geometric = False if geometric is None else geometric
        atmosphere = LIBREF_ATM if atmosphere is None else atmosphere

        # Only broadcast these parameters
        Mach, altitude = np.broadcast_arrays(Mach, altitude)
        return Mach, altitude, geometric, atmosphere

    def Tlapse(self, Mach: Hint.nums, altitude: Hint.nums = None,
               geometric: bool = None, atmosphere: object = None) -> np.ndarray:
        """
        Return the thrust lapse as a function of Mach number and altitude.

        Args:
            Mach: Flight Mach number.
            altitude: Flight altitude. Optional, defaults to altitude=0
                (sea-level conditions).
            geometric: Flag specifying if given altitudes are geometric or not.
                Optional, defaults to False.
            atmosphere: Atmosphere object. Optional, defaults to the library
                reference atmosphere.

        Returns:
            Thrust lapse parameter, Tlapse.

        """
        # Recast as necessary
        Mach, altitude, geometric, atmosphere = self._parse_arguments(
            Mach=Mach, altitude=altitude, geometric=geometric,
            atmosphere=atmosphere
        )

        # Compute
        Tlapse = self._f_Tlapse(
            Mach=Mach, altitude=altitude, geometric=geometric,
            atmosphere=atmosphere
        )
        return Tlapse

    def TSFC(self, Mach: Hint.nums, altitude: Hint.nums = None,
             geometric: bool = None, atmosphere: object = None) -> Quantity:
        """
        Return the installed thrust specific fuel consumption as a function of
        Mach number and altitude.

        Args:
            Mach: Flight Mach number.
            altitude: Flight altitude. Optional, defaults to altitude=0
                (sea-level conditions).
            geometric: Flag specifying if given altitudes are geometric or not.
                Optional, defaults to False.
            atmosphere: Atmosphere object. Optional, defaults to the library
                reference atmosphere.

        Returns:
            (Installed) Thrust-specific fuel consumption.

        """
        # Recast as necessary
        Mach, altitude, geometric, atmosphere = self._parse_arguments(
            Mach=Mach, altitude=altitude, geometric=geometric,
            atmosphere=atmosphere
        )

        # Compute
        TSFC = self._f_TSFC(
            Mach=Mach, altitude=altitude, geometric=geometric,
            atmosphere=atmosphere
        )
        return TSFC


# ---------------------------------------------------------------------------- #
# J. D. Mattingly - Basic decks for conceptual analysis

class MattinglyBasicTurbomachine(object):
    """Common methods for Mattingly's basic turbomachine models."""

    def __init__(self, TR: Hint.num = None):
        """
        Args:
            TR: Throttle ratio, a.k.a theta break of the engine.
        """
        self.TR = 1.05 if TR is None else TR
        return

    @property
    def TR(self) -> float:
        """
        Returns:
            The throttle ratio, a.k.a theta break (theta0 break).

        This is the stagnation temperature ratio at which maximum allowable
        compressor pressure ratio and turbine entry temperature are achieved
        simultaneously.

        """
        return self._TR

    @TR.setter
    def TR(self, value):
        self._TR = float(value)
        return


class HiBPR(EngineDeck, MattinglyBasicTurbomachine):
    """
    Performance deck for a high bypass ratio, subsonic turbofan.

    References:
        -   J. D. Mattingly, W. H. Heiser, D. T. Pratt, *Aircraft Engine Design*
            2nd ed. Reston, Virginia: AIAA, 2002. Sections 2.3.2, 3.3.2.

    """

    def _f_Tlapse(self, Mach: np.ndarray, altitude: np.ndarray,
                  geometric: np.ndarray[bool], atmosphere,
                  TR: Hint.nums = None) -> np.ndarray:
        # Recast as necessary
        TR = cast2numpy(self.TR if TR is None else TR)

        # Remove Mach numbers >= 0.9 from consideration
        warnmsg = f"{type(self).__name__} shouldn't be evaluated w/ Mach >= 0.9"
        if (Mach >= 0.9).any():
            warnings.warn(message=warnmsg, category=RuntimeWarning)
            Mach[Mach >= 0.9] = np.nan

        # Non-dimensional static and stagnation quantities
        theta, delta = TPratio(
            altitude=altitude,
            geometric=geometric,
            atmosphere=atmosphere
        )
        gamma = atmosphere.gamma(altitude=altitude, geometric=geometric)
        theta0 = theta * (Tt_T := 1 / IFlow.T_Tt(M=Mach, gamma=gamma))
        delta0 = delta * Tt_T ** (gamma / (gamma - 1))

        # Compute lapse
        Tlapse = delta0 * (1 - 0.49 * Mach ** 0.5)
        slice = theta0 > TR
        Tlapse[slice] -= (delta0 * (3 * (theta0 - TR) / (1.5 + Mach)))[slice]

        Tlapse[Tlapse < 0] = np.nan

        return Tlapse

    def _f_TSFC(self, Mach: np.ndarray, altitude: np.ndarray,
                geometric: np.ndarray[bool], atmosphere) -> Quantity:
        # Remove Mach numbers >= 0.9 from consideration
        warnmsg = f"{type(self).__name__} should not be evaluated at Mach > 0.9"
        if (Mach >= 0.9).any():
            warnings.warn(message=warnmsg, category=RuntimeWarning)
        Mach[Mach >= 0.9] = np.nan

        # Non-dimensional static quantity
        theta, _ = TPratio(
            altitude=altitude,
            geometric=geometric,
            atmosphere=atmosphere
        )

        # Compute TSFC
        TSFC = Quantity(  # Mattingly cancels lb with lbf, we do not cancel!
            (0.45 + 0.54 * Mach) * theta ** 0.5, "lb lbf^{-1} hr^{-1}")
        return TSFC


class LoBPRmixed(EngineDeck, MattinglyBasicTurbomachine):
    """
    Performance deck for a low bypass ratio, mixed flow turbofan.

    References:
    -   J. D. Mattingly, W. H. Heiser, D. T. Pratt, *Aircraft Engine Design*
        2nd ed. Reston, Virginia: AIAA, 2002. Sections 2.3.2, 3.3.2.

    """

    def _f_Tlapse(self, Mach: np.ndarray, altitude: np.ndarray,
                  geometric: np.ndarray[bool], atmosphere,
                  TR: Hint.nums = None) -> np.ndarray:
        # Recast as necessary
        TR = cast2numpy(self.TR if TR is None else TR)

        # Non-dimensional static and stagnation quantities
        theta, delta = TPratio(
            altitude=altitude,
            geometric=geometric,
            atmosphere=atmosphere
        )
        gamma = atmosphere.gamma(altitude=altitude, geometric=geometric)
        theta0 = theta * (Tt_T := 1 / IFlow.T_Tt(M=Mach, gamma=gamma))
        delta0 = delta * Tt_T ** (gamma / (gamma - 1))

        # Compute lapse
        Tlapse = 0.6 * delta0
        slice = theta0 > TR
        Tlapse[slice] *= (1 - 3.8 * (theta0 - TR) / theta0)[slice]

        Tlapse[Tlapse < 0] = np.nan

        return Tlapse

    @staticmethod
    def _f_TSFC(Mach: np.ndarray, altitude: np.ndarray,
                geometric: np.ndarray[bool], atmosphere) -> Quantity:
        # Non-dimensional static quantity
        theta, _ = TPratio(
            altitude=altitude,
            geometric=geometric,
            atmosphere=atmosphere
        )

        # Compute TSFC
        TSFC = Quantity(  # Mattingly cancels lb with lbf, we do not cancel!
            (0.9 + 0.30 * Mach) * theta ** 0.5, "lb lbf^{-1} hr^{-1}")
        return TSFC


class LoBPRmixedAB(EngineDeck, MattinglyBasicTurbomachine):
    """
    Performance deck for a low bypass ratio, mixed flow turbofan with reheat
    (afterburner) engaged.

    References:
    -   J. D. Mattingly, W. H. Heiser, D. T. Pratt, *Aircraft Engine Design*
        2nd ed. Reston, Virginia: AIAA, 2002. Sections 2.3.2, 3.3.2.

    """

    def _f_Tlapse(self, Mach: np.ndarray, altitude: np.ndarray,
                  geometric: np.ndarray[bool], atmosphere,
                  TR: Hint.nums = None) -> np.ndarray:
        # Recast as necessary
        TR = cast2numpy(self.TR if TR is None else TR)

        # Non-dimensional static and stagnation quantities
        theta, delta = TPratio(
            altitude=altitude,
            geometric=geometric,
            atmosphere=atmosphere
        )
        gamma = atmosphere.gamma(altitude=altitude, geometric=geometric)
        theta0 = theta * (Tt_T := 1 / IFlow.T_Tt(M=Mach, gamma=gamma))
        delta0 = delta * Tt_T ** (gamma / (gamma - 1))

        # Compute lapse
        Tlapse = delta0
        slice = theta0 > TR
        Tlapse[slice] *= (1 - 3.5 * (theta0 - TR) / theta0)[slice]

        Tlapse[Tlapse < 0] = np.nan

        return Tlapse

    @staticmethod
    def _f_TSFC(Mach: np.ndarray, altitude: np.ndarray,
                geometric: np.ndarray[bool], atmosphere) -> Quantity:
        # Non-dimensional static quantity
        theta, _ = TPratio(
            altitude=altitude,
            geometric=geometric,
            atmosphere=atmosphere
        )

        # Compute TSFC
        TSFC = Quantity(  # Mattingly cancels lb with lbf, we do not cancel!
            (1.6 + 0.27 * Mach) * theta ** 0.5, "lb lbf^{-1} hr^{-1}")
        return TSFC


class Turbojet(EngineDeck, MattinglyBasicTurbomachine):
    """
    Performance deck for a turbojet.

    References:
        -   J. D. Mattingly, W. H. Heiser, D. T. Pratt, *Aircraft Engine Design*
            2nd ed. Reston, Virginia: AIAA, 2002. Sections 2.3.2, 3.3.2.

    """

    def _f_Tlapse(self, Mach: np.ndarray, altitude: np.ndarray,
                  geometric: np.ndarray[bool], atmosphere,
                  TR: Hint.nums = None) -> np.ndarray:
        # Recast as necessary
        TR = cast2numpy(self.TR if TR is None else TR)

        # Non-dimensional static and stagnation quantities
        theta, delta = TPratio(
            altitude=altitude,
            geometric=geometric,
            atmosphere=atmosphere
        )
        gamma = atmosphere.gamma(altitude=altitude, geometric=geometric)
        theta0 = theta * (Tt_T := 1 / IFlow.T_Tt(M=Mach, gamma=gamma))
        delta0 = delta * Tt_T ** (gamma / (gamma - 1))

        # Compute lapse
        Tlapse = 0.8 * delta0 * (1 - 0.16 * Mach ** 0.5)
        slice = theta0 > TR
        Tlapse[slice] -= \
            (0.8 * delta0 * 24 * (theta0 - TR) / (9 + Mach) / theta0)[slice]

        Tlapse[Tlapse < 0] = np.nan

        return Tlapse

    @staticmethod
    def _f_TSFC(Mach: np.ndarray, altitude: np.ndarray,
                geometric: np.ndarray[bool], atmosphere) -> Quantity:
        # Non-dimensional static quantity
        theta, _ = TPratio(
            altitude=altitude,
            geometric=geometric,
            atmosphere=atmosphere
        )

        # Compute TSFC
        TSFC = Quantity(  # Mattingly cancels lb with lbf, we do not cancel!
            (1.1 + 0.30 * Mach) * theta ** 0.5, "lb lbf^{-1} hr^{-1}")
        return TSFC


class TurbojetAB(EngineDeck, MattinglyBasicTurbomachine):
    """
    Performance deck for a turbojet with reheat (afterburner) engaged.

    References:
        -   J. D. Mattingly, W. H. Heiser, D. T. Pratt, *Aircraft Engine Design*
            2nd ed. Reston, Virginia: AIAA, 2002. Sections 2.3.2, 3.3.2.

    """

    def _f_Tlapse(self, Mach: np.ndarray, altitude: np.ndarray,
                  geometric: np.ndarray[bool], atmosphere,
                  TR: Hint.nums = None) -> np.ndarray:
        # Recast as necessary
        TR = cast2numpy(self.TR if TR is None else TR)

        # Non-dimensional static and stagnation quantities
        theta, delta = TPratio(
            altitude=altitude,
            geometric=geometric,
            atmosphere=atmosphere
        )
        gamma = atmosphere.gamma(altitude=altitude, geometric=geometric)
        theta0 = theta * (Tt_T := 1 / IFlow.T_Tt(M=Mach, gamma=gamma))
        delta0 = delta * Tt_T ** (gamma / (gamma - 1))

        # Compute lapse
        Tlapse = delta0 * (1 - 0.3 * (theta0 - 1) - 0.1 * Mach ** 0.5)
        slice = theta0 > TR
        Tlapse[slice] -= (delta0 * 1.5 * (theta0 - TR) / theta0)[slice]

        Tlapse[Tlapse < 0] = np.nan

        return Tlapse

    @staticmethod
    def _f_TSFC(Mach: np.ndarray, altitude: np.ndarray,
                geometric: np.ndarray[bool], atmosphere) -> Quantity:
        # Non-dimensional static quantity
        theta, _ = TPratio(
            altitude=altitude,
            geometric=geometric,
            atmosphere=atmosphere
        )

        # Compute TSFC
        TSFC = Quantity(  # Mattingly cancels lb with lbf, we do not cancel!
            (1.5 + 0.23 * Mach) * theta ** 0.5, "lb lbf^{-1} hr^{-1}")
        return TSFC


class Turboprop(EngineDeck, MattinglyBasicTurbomachine):
    """
    Performance deck for a turboprop engine.

    References:
        -   J. D. Mattingly, W. H. Heiser, D. T. Pratt, *Aircraft Engine Design*
            2nd ed. Reston, Virginia: AIAA, 2002. Sections 2.3.2, 3.3.2.

    """

    def _f_Tlapse(self, Mach: np.ndarray, altitude: np.ndarray,
                  geometric: np.ndarray[bool], atmosphere,
                  TR: Hint.nums = None) -> np.ndarray:
        # Recast as necessary
        TR = cast2numpy(self.TR if TR is None else TR)

        # Non-dimensional static and stagnation quantities
        theta, delta = TPratio(
            altitude=altitude,
            geometric=geometric,
            atmosphere=atmosphere
        )
        gamma = atmosphere.gamma(altitude=altitude, geometric=geometric)
        theta0 = theta * (Tt_T := 1 / IFlow.T_Tt(M=Mach, gamma=gamma))
        delta0 = delta * Tt_T ** (gamma / (gamma - 1))

        # Compute lapse
        Tlapse = delta0
        slice0 = Mach > 0.1
        slice1 = slice0 & (theta0 > TR)
        # I think Mattingly makes a mistake below in writing M-1 instead of M-.1
        Tlapse[slice0] *= (1 - 0.96 * (Mach - 0.1) ** 0.25)[slice0]
        Tlapse[slice1] -= \
            (delta0 * 3 * (theta0 - TR) / 8.13 / (Mach - 0.1))[slice1]

        Tlapse[Tlapse < 0] = np.nan

        return Tlapse

    @staticmethod
    def _f_TSFC(Mach: np.ndarray, altitude: np.ndarray,
                geometric: np.ndarray[bool], atmosphere) -> Quantity:
        # Non-dimensional static quantity
        theta, _ = TPratio(
            altitude=altitude,
            geometric=geometric,
            atmosphere=atmosphere
        )

        # Compute TSFC
        TSFC = Quantity(  # Mattingly cancels lb with lbf, we do not cancel!
            (0.18 + 0.8 * Mach) * theta ** 0.5, "lb lbf^{-1} hr^{-1}")
        return TSFC


# Collect all the basic Mattingly decks
mattinglybasicdecks = {
    HiBPR.__name__: HiBPR,
    LoBPRmixed.__name__: LoBPRmixed,
    LoBPRmixedAB.__name__: LoBPRmixedAB,
    Turbojet.__name__: Turbojet,
    TurbojetAB.__name__: TurbojetAB,
    Turboprop.__name__: Turboprop
}


# Define a catalogue of basic decks from Mattingly
class MattinglyBasic(type("catalogue", (object,), mattinglybasicdecks)):
    """
    A collection of algebraic equations to model engine installed thrust lapse
    and installed thrust specific fuel consumption from year 2000 and beyond.

    References:
    -   J. D. Mattingly, W. H. Heiser, D. T. Pratt, *Aircraft Engine Design*
        2nd ed. Reston, Virginia: AIAA, 2002. Sections 2.3.2, 3.3.2.

    """

    def __init__(self):
        errormsg = (
            "This is a catalogue of atmospheres, and it should not be "
            "instantiated directly. Try one of my attributes!"
        )
        raise RuntimeError(errormsg)
