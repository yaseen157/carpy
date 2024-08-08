"""A module implementing the methods required to model celestial objects, coordinate frames, and kinematic behaviour."""
import numpy as np

from carpy.environment.navigation._ellipsoids import ReferenceEllipsoid, WGS84
from carpy.environment.navigation._gravfields import GravFieldModel, EGM96s

__all__ = ["CelestialBody", "Earth"]


class CelestialBody:

    def __init__(self, ellipsoid: ReferenceEllipsoid, gravitational_field_model: GravFieldModel):
        self._ellipsoid = ellipsoid
        self._gravfield = gravitational_field_model
        return

    @property
    def ellipsoid(self) -> ReferenceEllipsoid:
        """Idealised geodetic reference for height, intended to approximate the body's form."""
        return self._ellipsoid

    @property
    def gravitational_field_model(self) -> GravFieldModel:
        """Computational model for the gravitational field."""
        return self._gravfield

    def potential(self, lat, lon, alt):
        """
        Given the geographic/planetodetic spatial ellipsoidal coordinates (phi, lambda, h), compute and return the
        sum of (gravitational) attraction and centrifugal potentials.

        Args:
            lat: Spatial ellipsoidal coordinate phi, representing geographic latitude, in radians.
            lon: Spatial ellipsoidal coordinate lambda, representing geographic longitude, in radians.
            alt: Geometric height above the ellipsoid, h.

        Returns:
            Total potential.

        """
        # Compute spherical coordinate system parameters
        x, y, z = self.ellipsoid.lla_to_xyz(lat=lat, lon=lon, alt=alt)
        rad = (x ** 2 + y ** 2 + z ** 2) ** 0.5
        lat_bar = np.arctan2(z, (x ** 2 + y ** 2) ** 0.5)

        # Compute potential
        attractive_potential = self.gravitational_field_model.attraction_potential(rad=rad, lon=lon, lat=lat_bar)
        centrifugal_potential = self.ellipsoid.centrifugal_potential(lat=lat, lon=lon, alt=alt)

        # TODO: Figure out a nice way of eliminating "units" of radians and steradians when operations are done to the
        #   Quantity that involve length dimensions. For example, F = m r omega with units [Pa] = [kg] [m] [rad s^-1]
        #   should see that the radian, which almost describes an absence of length dimensions, should disappear when
        #   multiplied by a length dimension. Similarly, lengths shouldn't cancel if operated on by arcsin, arccos,
        #   arctan without creating radians in the process.
        potential = attractive_potential + centrifugal_potential

        return potential


class Earth(CelestialBody):

    def __init__(self, ellipsoid: ReferenceEllipsoid = None, gravitational_field_model=None):
        """
        Args:
            ellipsoid: Reference ellipsoidal model. Optional, defaults to the WGS84 oblate spheroid.
        """
        # Recast as necessary
        ellipsoid = WGS84() if ellipsoid is None else ellipsoid
        gravitational_field_model = EGM96s() if gravitational_field_model is None else gravitational_field_model

        # Superclass call
        super().__init__(ellipsoid=ellipsoid, gravitational_field_model=gravitational_field_model)
        return


if __name__ == "__main__":
    planet = Earth()
    print(planet.potential(np.radians([[0], [1]]), 0, 0))
