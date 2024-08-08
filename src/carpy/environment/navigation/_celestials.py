"""A module implementing the methods required to model celestial objects, coordinate frames, and kinematic behaviour."""
from carpy.environment.navigation._ellipsoids import ReferenceEllipsoid, WGS84


class CelestialBody:

    def __init__(self, ellipsoid: ReferenceEllipsoid):
        self._ellipsoid = ellipsoid
        return

    @property
    def ellipsoid(self) -> ReferenceEllipsoid:
        """Idealised geodetic reference for height, intended to approximate the body's form."""
        return self._ellipsoid


class Earth(CelestialBody):

    def __init__(self, ellipsoid: ReferenceEllipsoid = None):
        """
        Args:
            ellipsoid: Reference ellipsoidal model. Optional, defaults to the WGS84 oblate spheroid.
        """
        # Recast as necessary
        ellipsoid = WGS84() if ellipsoid is None else ellipsoid

        # Superclass call
        super().__init__(ellipsoid=ellipsoid)
        return
