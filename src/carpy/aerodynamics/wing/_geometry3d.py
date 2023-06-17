"""Methods for modelling the 3d geometry of a wing."""
import warnings

import numpy as np
from scipy.integrate import simpson as sint_simpson

from carpy.aerodynamics.aerofoil import BaseProfile
from carpy.utility import Hint, Quantity

__all__ = ["Wings"]
__author__ = "Yaseen Reza"


class Station(object):
    """
    Simple class for recording geometrical parameters of wing stations.

    Currently, the chordwise station depicting centre of geometric twist is not
    included in this representation.
    """

    def __init__(self, profile: BaseProfile, incidence: float, dihedral: float):
        self._profile = profile
        self._incidence = incidence
        self._dihedral = dihedral
        return

    @property
    def profile(self) -> BaseProfile:
        """Profile object describing the 2D cross-section of a wing."""
        return self._profile

    @property
    def incidence(self) -> float:
        """Angle of incidence of the wing station."""
        return self._incidence

    @property
    def dihedral(self) -> float:
        """
        Angle of dihedral of the section bounded by this station and the one
        inboard of it.
        """
        return self._dihedral


class Monoplane(object):
    """Class for modelling monoplane wings."""
    _stations: dict

    def __init__(self, planform: object):
        """
        Args:
            planform (Wing Planform object): Two-dimensional projected planform
                of the wing.
        """
        self._stations = dict()
        self._planform = planform
        return

    @property
    def planform(self):
        """Two-dimensional projected planform of the wing."""
        return self._planform

    def set_station(self, eta: Hint.num, profile: BaseProfile,
                    incidence: Hint.num = None,
                    dihedral: Hint.num = None) -> None:
        """
        Define the geometry of the wing at a given station along the span.

        Args:
            eta: The non-dimensional position of the station along the semispan.
                Can take values in the domain [0, 1] representing port and
                starboard directions simultaneously (symmetrical wing).
            profile (Aerofoil Profile object): Two-dimensional profile of the
                representing the geometry of the aerofoil at this station.
            incidence: The absolute angle the chord line of the aerofoil
                profile makes with the reference axis along the fuselage.
                Optional, defaults to 0 degrees (no incidence angle).
            dihedral: The absolute angle of the zero-incidence section bounded
                by this station and the one inboard of it with the horizontal
                plane. Optional, defaults to 0 degrees (no dihedral angle).

        Returns:
            None.

        Notes:
            A position of twist *should* be identified for the most rigourous
                definition of the wing's geometry. It does not affect the
                computations of the wing's wetted area or internal volume.

        """
        # Recast as necessary
        dihedral = 0 if dihedral is None else dihedral
        incidence = 0 if incidence is None else incidence
        eta_stbd = eta

        # Starboard/port station (symmetrical)
        if eta in self._stations:
            warnings.warn(f"Overwriting station at {eta=}")

        self._stations[eta_stbd] = Station(
            profile=profile,
            incidence=incidence,
            dihedral=dihedral
        )

        return None

    def c(self, y: Hint.nums) -> Quantity:
        """
        Chord length of the wing.

        Args:
            y: Semi-span coordinate in the horizontal plane at which to evaluate
                the chord length of the wing.

        Returns:
            Chord length(s) of the wing at selected stations, linearly
                interpolated between the nearest available stations.

        """
        # Recast as necessary
        y = Quantity(y, "m")

        # Evaluate in the starboard direction.
        stations = {k: v for (k, v) in self._stations.items() if k >= 0}

        # If no stations have been defined, simply return the projected result
        if len(stations) == 0:
            return self.planform.c(y=y)

        # If stations have been defined, linearly interpolate chord lengths
        # based on an aerofoil with local twist and associated chord matching
        # the planform projection.
        nd_chordlengths = []

        for i, (eta, station) in enumerate(stations.items()):
            # Chord length of the station
            ctheta_i = np.cos(station.incidence)
            nd_chord_aerofoil = 1 / ctheta_i
            nd_chordlengths.append(nd_chord_aerofoil)

        # Linearly interpolate the answer between stations
        b_2 = self.planform.b / 2
        nd_c = np.interp((abs(y) / b_2).x, list(stations), nd_chordlengths)
        c = nd_c * self.planform.c(y=y)

        return c

    @property
    def Sw(self) -> Quantity:
        """Wetted area of the isolated wing."""
        # Evaluate in the starboard direction.
        stations = {k: v for (k, v) in self._stations.items() if k >= 0}

        # If no stations have been defined, simply return the projected result
        if len(stations) == 0:
            warnmsg = f"No stations have been defined, assuming flat-plate area"
            warnings.warn(message=warnmsg, category=RuntimeWarning)
            Sw = self.planform.S * 2  # Wet area is 2x the reference area
            return Sw

        # If stations have been defined, linearly interpolate the perimeters
        nd_perimeters = []

        for i, (eta, station) in enumerate(stations.items()):
            # Wetted cross-sectional perimeter of the station
            nd_perimeter = station.profile.nd_P
            nd_perimeters.append(nd_perimeter)

        # Linearly interpolate the answer between stations
        etas = np.linspace(0, 1, 100)
        nd_Ps = np.interp(etas, list(stations), nd_perimeters)
        # ...non-dimensional perimeters should dimensionalise through chord len.
        Ps = (nd_Ps * self.c(y=etas * self.planform.b / 2))
        # ...non-dimensional span (eta) should dimensionalise through planform b
        Sw = 2 * Quantity(sint_simpson(Ps, etas * self.planform.b / 2), "m^{2}")

        return Sw

    def view(self) -> tuple:
        """
        Compute the coordinates of and draw a 3D wing geometry in matplotlib.

        Returns:
            A tuple of figure and axes (subplot) objects from matplotlib.

        """
        # Create a figure and plot things
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        _, _ = fig, ax
        # X, Z = plane of aerofoils
        # then use incidence to rotate aerofoils
        # then use

        return NotImplemented, NotImplemented


class Wings(object):
    """A collection of (fixed-)wing configurations, packaged for easy access."""
    Monoplane = Monoplane
    # Biplane = NotImplemented
    # Triplane = NotImplemented
    # Cruciform = NotImplemented
