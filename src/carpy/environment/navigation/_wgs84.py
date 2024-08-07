"""Implementation of methods for locating objects in the World Geodetic System (1984)."""
import numpy as np

from carpy.utility import Quantity, constants as co

__all__ = ["WGS84"]
__author__ = "Yaseen Reza"


class EllipsoidalFixedFrame:
    _a: Quantity
    _b: Quantity

    def __init__(self, a, b):
        self._a = Quantity(a, "m")
        self._b = Quantity(b, "m")
        return

    @property
    def a(self) -> Quantity:
        """Ellipsoidal semi-major axis."""
        return self._a

    @property
    def b(self) -> Quantity:
        """Ellipsoidal semi-minor (typically polar) axis."""
        return self._b

    @property
    def e_sq(self):
        """Square of the eccentricity parameter of the ellipsoid."""
        one_minus_f = (self.b / self.a)
        e_sq = 1 - one_minus_f ** 2
        return e_sq

    def N_curvature(self, lat):
        # Prime vertical radius of curvature (function of phi)
        N_phi = self.a / (1 - self.e_sq * np.sin(lat) ** 2) ** 0.5
        return N_phi

    def M_curvature(self, lat):
        # Rename variables for convenience
        e_sq = self.e_sq
        N_phi = self.N_curvature(lat=lat)

        # Meridian radius of curvature
        M_phi = N_phi * (1 - e_sq) / (1 - e_sq * np.sin(lat) ** 2)
        return M_phi

    @staticmethod
    def enu_to_ned(e_hat, n_hat, u_hat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given unit direction vectors for an East North Up (ENU) frame, return the North East Down (NED) unit vectors.

        Args:
            e_hat: East-directing unit vector.
            n_hat: North-directing unit vector.
            u_hat: Up-directing unit vector.

        Returns:
            Unit vectors for local North, East, and Down directions of a planetocentric reference frame.

        """
        # If a compass in your hand has a north pointing, east pointing, and up pointing vector, a 180 degree rotation
        #   around the NE direction swaps N and E, and flips the sign of U (to D).
        return n_hat, e_hat, -u_hat

    @classmethod
    def ned_to_enu(cls, n_hat, e_hat, d_hat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given unit direction vectors for an East North Up (ENU) frame, return the North East Down (NED) unit vectors.

        Args:
            n_hat: North-directing unit vector.
            e_hat: East-directing unit vector.
            d_hat: Down-directing unit vector.

        Returns:
            Unit vectors for local East, North, and Up directions of a planetocentric reference frame.

        """
        # As described in the other function, enu to ned is 180 degrees rotation around ne. It follows that ned to enu
        #   is also 180 degrees about ne!
        return cls.enu_to_ned(e_hat=n_hat, n_hat=e_hat, u_hat=d_hat)

    @staticmethod
    def orient_enu(lat, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return unit vectors e_hat, n_hat, and u_hat defining the orientation of the EastNorthUp (ENU) coordinate system.

        Each unit vector is the row of a 3x3 rotation matrix R such that [e n u] = R @ [x y z], and d[ENU] = R @ d[XYZ].

        Args:
            lat: Ellipsoidal latitude, in radians.
            lon: Ellipsoidal longitude, in radians.

        Returns:
            Unit vectors for local East, North, and Up directions of a planetocentric reference frame.

        Notes:
            If latitude and longitude are ellipsoidal coordinates, u_hat is orthogonal to the tangent plane to the
            ellispoid. If latitude and longitude are spherical, the vector u_hat is now orthogonal to the tangent plane
            of a sphere.

        References:
            https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates

        """
        # Pre-compute trigonometry
        sin_phi, cos_phi = np.sin(lat), np.cos(lat)
        sin_lmd, cos_lmd = np.sin(lon), np.cos(lon)

        e_hat = np.array([-sin_lmd, cos_lmd, 0])
        n_hat = np.array([-cos_lmd * sin_phi, -sin_lmd * sin_phi, cos_phi])
        u_hat = np.array([cos_lmd * cos_phi, sin_lmd * cos_phi, sin_phi])

        return e_hat, n_hat, u_hat

    @classmethod
    def orient_xyz(cls, lat, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return unit vectors x_hat, y_hat, and z_hat defining the orientation of the planetocentric coordinate system.

        Each unit vector is the row of a 3x3 rotation matrix R such that [x y z] = R @ [e n u], and d[XYZ] = R @ d[ENU].

        Args:
            lat: Ellipsoidal latitude, in radians.
            lon: Ellipsoidal longitude, in radians.

        Returns:
            Unit vectors for X, Y, and Z directions of a planetocentric reference frame.

        References:
            https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates

        """
        # x hat, y hat, and z hat are simply the rows of the transposed e hat, n hat, and u hat rotation matrix.
        x_hat, y_hat, z_hat = np.vstack(cls.orient_enu(lat=lat, lon=lon)).T
        return x_hat, y_hat, z_hat

    def _dENU(self, lat, lon, alt, dlat, dlon, dalt):
        """
        Differential in ENU frame for small changes in planetodetic frame.

        References:
            https://en.wikipedia.org/wiki/Geographic_coordinate_conversion

        """
        mat = np.array([(self.N_curvature(lat=lat) + alt) * np.cos(lat), self.M_curvature(lat - 0), 1]) * np.identity(3)
        dENU = mat @ np.array([[dlon], [dlat], [dalt]])
        return dENU


class WGS84(EllipsoidalFixedFrame):

    def __init__(self):
        super().__init__(a=co.STANDARD.WGS84.a, b=co.STANDARD.WGS84.b)
        return

    def _planetodetic_to_XYZ(self, lat, lon, alt) -> tuple[float, float, float]:
        """
        Return position vector in planetocentric space R = (X, Y, Z) given planetodetic coordinates.

        Args:
            lat: Ellipsoidal latitude, in radians.
            lon: Ellipsoidal longitude, in radians.
            alt: Geometric altitude above the ellipsoid surface.

        Returns:
            Position vector in Earth-centred Earth-fixed coordinate system.

        References:
            https://en.wikipedia.org/wiki/Geographic_coordinate_conversion

        """
        # Pre-compute trigonometry
        sin_phi, cos_phi = np.sin(lat), np.cos(lat)
        sin_lmd, cos_lmd = np.sin(lon), np.cos(lon)

        # Prime vertical radius of curvature (function of phi)
        N_phi = self.N_curvature(lat=lat)

        X = (N_phi + alt) * cos_phi * cos_lmd
        Y = (N_phi + alt) * cos_phi * sin_lmd
        Z = ((1 - self.e_sq) * N_phi + alt) * sin_phi

        return X, Y, Z


if __name__ == "__main__":
    lats = np.linspace(-np.pi, np.pi, 30)
    lons = np.linspace(-np.pi, np.pi, 30)

    LATS, LONS = np.meshgrid(lats, lons)

    X, Y, Z = WGS84()._planetodetic_to_XYZ(lat=LATS, lon=LONS, alt=0)

    import pyvista as pv

    points = np.vstack((X.flat, Y.flat, Z.flat)).T

    mesh = pv.PolyData(points)
    mesh.plot(point_size=10, style="points")
