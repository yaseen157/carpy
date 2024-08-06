"""Implementation of methods for locating objects in the World Geodetic System (1984)."""
import numpy as np

from carpy.utility import constants as co

__all__ = ["WGS84"]
__author__ = "Yaseen Reza"


class WGS84(type("BodyFixedFrame", (object,), {})):

    def _planetodetic_to_planetocentric(self, lat, lon, h) -> tuple[float, float, float]:
        """
        Return position vector in space R = (X, Y, Z) given planetodetic coordinates.

        Args:
            lat: Planetodetic latitude, in radians.
            lon: Planetodetic longitude, in radians.
            h: Height above the ellipsoid surface.

        Returns:
            Position vector in Earth-centred Earth-fixed coordinate system.

        References:
            https://en.wikipedia.org/wiki/Geographic_coordinate_conversion

        """
        # Rename variables for convenience
        a = co.STANDARD.WGS84.a
        b = co.STANDARD.WGS84.b
        one_minus_f = (b / a)
        e_sq = 1 - (one_minus_f) ** 2

        # Pre-compute trigonometry
        sin_phi, cos_phi = np.sin(lat), np.cos(lat)
        sin_lmd, cos_lmd = np.sin(lon), np.cos(lon)

        # Prime vertical radius of curvature (function of phi)
        N_phi = a / (1 - e_sq * sin_phi ** 2) ** 0.5

        X = (N_phi + h) * cos_phi * cos_lmd
        Y = (N_phi + h) * cos_phi * sin_lmd
        Z = ((1 - e_sq) * N_phi + h) * sin_phi

        return X, Y, Z

    def _planetodetic_to_enu(self, lat, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return EastNorthUp (ENU) coordinate system orientation, given planetodetic parameters.

        Args:
            lat: Planetodetic latitude, in radians.
            lon: Planetodetic longitude, in radians.

        Returns:
            Unit vectors for local East, North, and Up directions of a planetocentric reference frame.

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

    def _enu_to_ned(self, e_hat, n_hat, u_hat):
        """
        Given unit direction vectors for an East North Up (ENU) frame, return the North East Down (NED) unit vectors.

        Args:
            e_hat: East-directing unit vector.
            n_hat: North-directing unit vector.
            u_hat: Up-directing unit vector.

        Returns:
            Unit vectors for local North, East, and Down directions of a planetocentric reference frame.

        References:
            https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates

        """
        return n_hat, e_hat, -u_hat

    def _planetodetic_to_xyz(self, lat, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return unit vectors in planetocentric directions, given coordinates in EastNorthUp (ENU).

        Args:
            lat: Latitude, in radians.
            lon: Longitude, in radians.

        Returns:
            Unit vectors for local X, Y, and Z directions of a planetocentric reference frame.

        References:
            https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates

        """
        # Pre-compute trigonometry
        sin_phi, cos_phi = np.sin(lat), np.cos(lat)
        sin_lmd, cos_lmd = np.sin(lon), np.cos(lon)

        x_hat = np.array([-sin_lmd, -cos_lmd * sin_phi, cos_lmd * cos_phi])
        y_hat = np.array([cos_lmd, -sin_lmd * sin_lmd, cos_phi])
        z_hat = np.array([0, cos_phi, sin_phi])

        return x_hat, y_hat, z_hat
