"""A module for defining mission environments, including gravitational and atmospheric parameters."""
import datetime as dt
import numpy as np

from carpy.environment import atmospheres as atms, planetodetics as planet

__all__ = ["Environment", "Earth"]


class Environment:
    _planet_ellipsoid: planet.ReferenceEllipsoid
    _planet_gravfield: planet.GravFieldModel
    _static_atmosphere: atms.StaticAtmosphereModel

    @property
    def planet_ellipsoid(self) -> planet.ReferenceEllipsoid:
        """The reference ellipsoid, or ellipsoid of revolution approximating a planet's form."""
        return self._planet_ellipsoid

    @property
    def planet_gravfield(self) -> planet.GravFieldModel:
        """A spatially defined gravitational field strength model."""
        return self._planet_gravfield

    @property
    def static_atmosphere(self) -> atms.StaticAtmosphereModel:
        """A simple altitude based, mission relevant reference atmosphere."""
        return self._static_atmosphere

    def gravity_potential(self, lat, lon, alt):
        """
        Given the geographic/planetodetic spatial ellipsoidal coordinates (phi, lambda, h), compute and return the
        sum of gravitational attraction and centrifugal potentials.

        Args:
            lat: Spatial ellipsoidal coordinate phi, representing geographic latitude, in radians.
            lon: Spatial ellipsoidal coordinate lambda, representing geographic longitude, in radians.
            alt: Geometric height above the ellipsoid, h.

        Returns:
            Total potential.

        """
        # Compute spherical coordinate system parameters
        x, y, z = self.planet_ellipsoid.lla_to_xyz(lat=lat, lon=lon, alt=alt)
        rad = (x ** 2 + y ** 2 + z ** 2) ** 0.5
        lat_bar = np.arctan2(z, (x ** 2 + y ** 2) ** 0.5)

        # Compute potential
        attractive_potential = self.planet_gravfield.attraction_potential(rad=rad, lon=lon, lat=lat_bar)
        centrifugal_potential = self.planet_ellipsoid.centrifugal_potential(lat=lat, lon=lon, alt=alt)
        potential = attractive_potential + centrifugal_potential

        return potential


class Earth(Environment):
    """
    A reference Earth model for the easy instantiation of an Earth environment.

    The model defaults to:
    -   WGS84 Earth reference ellipsoid.
    -   EGM96s Earth gravitational model.
    -   ISO 2533:1975 International Standard Atmosphere.

    """

    def __init__(self):
        self._planet_ellipsoid = planet.WGS84()
        self._planet_gravfield = planet.EGM96s()
        self._static_atmosphere = atms.ISO_2533_1975()
        return

# TODO: Permit user definition of a planetodetic epoch, and the parameters describing solar cycles with datetime
# TODO: Permit user definition of "static" (invariant) diurnal cycles, allowing for "worst case" designs
