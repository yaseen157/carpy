"""A module implementing a number of models for reference ellipsoid shapes, and parameters of their transformations."""
import warnings

import numpy as np

from carpy.utility import Quantity, constants as co

__all__ = ["ReferenceEllipsoid", "WGS84"]
__author__ = "Yaseen Reza"


class ReferenceEllipsoid:
    """Base class providing methods to fixed-frame ellipsoids"""
    _GM: Quantity
    _a: Quantity
    _b: Quantity
    _omega: Quantity

    def __init__(self, GM, a, b, omega):
        self._GM = Quantity(GM, "m^3 s^-2")
        self._a = Quantity(a, "m")
        self._b = Quantity(b, "m")
        self._omega = Quantity(omega, "rad s^{-1}")
        return

    @property
    def GM(self) -> Quantity:
        """Standard gravitational parameter. Computed as the product of the gravitational constant G and body mass M."""
        return self._GM

    @property
    def a(self) -> Quantity:
        """Ellipsoidal semi-major (equatorial) axis."""
        return self._a

    @property
    def b(self) -> Quantity:
        """Ellipsoidal semi-minor (polar) axis."""
        return self._b

    @property
    def omega(self) -> Quantity:
        """Angular velocity of the Earth's rotation."""
        return self._omega

    @property
    def e_sq(self):
        """Square of the first eccentricity parameter."""
        e_sq = self.f * (2 - self.f)
        return e_sq

    @property
    def E(self):
        """Linear eccentricity."""
        return (self.a ** 2 - self.b ** 2) ** 0.5

    @property
    def e(self):
        """First numerical eccentricity."""
        return self.E / self.a

    @property
    def e_prime(self):
        """Second numerical eccentricity."""
        return self.E / self.b

    @property
    def f(self):
        """Flattening parameter."""
        f = (self.a - self.b) / self.a
        return f

    def nu(self, lat):
        # Prime vertical radius of curvature (function of phi)
        N_phi = self.a / (1 - self.e_sq * np.sin(lat) ** 2) ** 0.5
        return N_phi

    def mu(self, lat):
        # Rename variables for convenience
        e_sq = self.e_sq
        nu_phi = self.nu(lat=lat)

        # Meridian radius of curvature
        mu_phi = nu_phi * (1 - e_sq) / (1 - e_sq * np.sin(lat) ** 2)
        return mu_phi

    def lla_to_xyz(self, lat, lon, alt):
        """
        Given the geographic/planetodetic spatial ellipsoidal coordinates (phi, lambda, h), compute and return the
        equivalent position in rectangular coordinates (x, y, z).

        Args:
            lat: Spatial ellipsoidal coordinate phi, representing geographic latitude, in radians.
            lon: Spatial ellipsoidal coordinate lambda, representing geographic longitude, in radians.
            alt: Geometric height above the ellipsoid, h.

        Returns:
            A tuple of X, Y, and Z positions.

        References:
            https://www.icao.int/NACC/Documents/Meetings/2014/ECARAIM/REF08-Doc9674.pdf, Appendix D-1.

        """
        # Pre-compute trigonometry
        sin_phi, cos_phi = np.sin(lat), np.cos(lat)
        sin_lmd, cos_lmd = np.sin(lon), np.cos(lon)

        # Prime vertical radius of curvature (function of phi)
        nu_phi = self.nu(lat=lat)

        x = (nu_phi + alt) * cos_phi * cos_lmd
        y = (nu_phi + alt) * cos_phi * sin_lmd
        z = (nu_phi * (1 - self.e_sq) + alt) * sin_phi

        return x, y, z

    def xyz_to_lla(self, x, y, z):
        """
        Transform from the cartesian coordinates (x, y, z) of the local system into the equivalent position in
        geographic/planetodetic spatial ellipsoidal coordinates (phi, lambda, h).

        Args:
            x: Position in cartesian space, x-aligned.
            y: Position in cartesian space, y-aligned.
            z: Position in cartesian space, z-aligned.

        Returns:
            A tuple of phi and lambda arguments, and the ellipsoid height h.

        References:
            WGS 84 Implementation Manual. International Civil Aviation Organisation, 1998. Accessed: Aug. 08, 2024.
                [Online]. Available: https://www.icao.int/safety/pbn/Documentation/EUROCONTROL/Eurocontrol%20WGS%2084%20Implementation%20Manual.pdf

        """
        x, y, z = np.broadcast_arrays(x, y, z, subok=True)

        r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
        if np.any(select := (r <= 43e3)):
            warn_msg = f"Cartesian coordinates within 43 kilometres of the Earth's centre cannot be mapped to ellipsoid"
            warnings.warn(message=warn_msg, category=RuntimeWarning)
            x[select] = np.nan
            y[select] = np.nan
            z[select] = np.nan

        w = (x ** 2 + y ** 2) ** 0.5
        l = self.e_sq / 2
        m = (w / self.a) ** 2
        n = ((1 - self.e ** 2) * z / self.b) ** 2
        i = -(2 * l ** 2 + m + n) / 2
        k = l ** 2 * (l ** 2 - m - n)
        mnl2 = m * n * l ** 2
        q = (m + n - 4 * l ** 2) ** 3 / 216 + mnl2
        D = ((2 * q - mnl2) * mnl2) ** 0.5
        beta = i / 3 - (q + D) ** (1 / 3) - (q - D) ** (1 / 3)
        t = ((beta ** 2 - k) ** 0.5 - (beta + i) / 2) ** 0.5 - np.sin(m - n) * ((beta - i) / 2) ** 0.5
        w1 = w / (t + l)
        z1 = (1 - self.e_sq) * z / (t - l)

        lat = np.arctan2(z1, ((1 - self.e_sq) * w1))
        lon = 2 * np.arctan2((w - x), y)
        alt = np.sin(t - 1 + l) * ((w - w1) ** 2 + (z - z1) ** 2) ** 0.5

        return lat, lon, alt

    def centrifugal_potential(self, lat, lon, alt):
        """
        Given the geographic/planetodetic spatial ellipsoidal coordinates (phi, lambda, h), compute and return the
        centrifugal potential.

        Args:
            lat: Spatial ellipsoidal coordinate phi, representing geographic latitude, in radians.
            lon: Spatial ellipsoidal coordinate lambda, representing geographic longitude, in radians.
            alt: Geometric height above the ellipsoid, h.

        Returns:
            Centrifugal potential.

        """
        # Compute distance to axis of rotation (z-axis)
        x, y, _ = self.lla_to_xyz(lat=lat, lon=lon, alt=alt)
        d_z = (x ** 2 + y ** 2) ** 0.5

        # Compute centrifugal potential
        Phi = 0.5 * (self.omega * d_z) ** 2
        return Phi

    # @staticmethod
    # def enu_to_ned(e_hat, n_hat, u_hat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Given unit direction vectors for an East North Up (ENU) frame, return the North East Down (NED) unit vectors.
    #
    #     Args:
    #         e_hat: East-directing unit vector.
    #         n_hat: North-directing unit vector.
    #         u_hat: Up-directing unit vector.
    #
    #     Returns:
    #         Unit vectors for local North, East, and Down directions of a planetocentric reference frame.
    #
    #     """
    #     # If a compass in your hand has a north pointing, east pointing, and up pointing vector, a 180 degree rotation
    #     #   around the NE direction swaps N and E, and flips the sign of U (to D).
    #     return n_hat, e_hat, -u_hat
    #
    # @classmethod
    # def ned_to_enu(cls, n_hat, e_hat, d_hat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Given unit direction vectors for an East North Up (ENU) frame, return the North East Down (NED) unit vectors.
    #
    #     Args:
    #         n_hat: North-directing unit vector.
    #         e_hat: East-directing unit vector.
    #         d_hat: Down-directing unit vector.
    #
    #     Returns:
    #         Unit vectors for local East, North, and Up directions of a planetocentric reference frame.
    #
    #     """
    #     # As described in the other function, enu to ned is 180 degrees rotation around ne. It follows that ned to enu
    #     #   is also 180 degrees about ne!
    #     return cls.enu_to_ned(e_hat=n_hat, n_hat=e_hat, u_hat=d_hat)
    #
    # @staticmethod
    # def orient_enu(lat, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Return unit vectors e_hat, n_hat, and u_hat defining the orientation of the EastNorthUp (ENU) coordinate system.
    #
    #     Each unit vector is the row of a 3x3 rotation matrix R such that [e n u] = R @ [x y z], and d[ENU] = R @ d[XYZ].
    #
    #     Args:
    #         lat: Ellipsoidal latitude, in radians.
    #         lon: Ellipsoidal longitude, in radians.
    #
    #     Returns:
    #         Unit vectors for local East, North, and Up directions of a planetocentric reference frame.
    #
    #     Notes:
    #         If latitude and longitude are ellipsoidal coordinates, u_hat is orthogonal to the tangent plane to the
    #         ellispoid. If latitude and longitude are spherical, the vector u_hat is now orthogonal to the tangent plane
    #         of a sphere.
    #
    #     References:
    #         https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    #
    #     """
    #     # Pre-compute trigonometry
    #     sin_phi, cos_phi = np.sin(lat), np.cos(lat)
    #     sin_lmd, cos_lmd = np.sin(lon), np.cos(lon)
    #
    #     e_hat = np.array([-sin_lmd, cos_lmd, 0])
    #     n_hat = np.array([-cos_lmd * sin_phi, -sin_lmd * sin_phi, cos_phi])
    #     u_hat = np.array([cos_lmd * cos_phi, sin_lmd * cos_phi, sin_phi])
    #
    #     return e_hat, n_hat, u_hat
    #
    # @classmethod
    # def orient_xyz(cls, lat, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Return unit vectors x_hat, y_hat, and z_hat defining the orientation of the planetocentric coordinate system.
    #
    #     Each unit vector is the row of a 3x3 rotation matrix R such that [x y z] = R @ [e n u], and d[XYZ] = R @ d[ENU].
    #
    #     Args:
    #         lat: Ellipsoidal latitude, in radians.
    #         lon: Ellipsoidal longitude, in radians.
    #
    #     Returns:
    #         Unit vectors for X, Y, and Z directions of a planetocentric reference frame.
    #
    #     References:
    #         https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    #
    #     """
    #     # x hat, y hat, and z hat are simply the rows of the transposed e hat, n hat, and u hat rotation matrix.
    #     x_hat, y_hat, z_hat = np.vstack(cls.orient_enu(lat=lat, lon=lon)).T
    #     return x_hat, y_hat, z_hat
    #
    # def _dENU(self, lat, lon, alt, dlat, dlon, dalt) -> tuple:
    #     """
    #     Differential in ENU frame for small changes in planetodetic frame.
    #
    #     References:
    #         https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
    #
    #     """
    #     dlon, dlat, dalt = map(np.atleast_1d, (dlon, dlat, dalt))
    #
    #     mat = np.array([(self.N_curvature(lat=lat) + alt) * np.cos(lat), self.M_curvature(lat - 0), 1]) * np.identity(3)
    #     dE, dN, dU = mat @ np.array([dlon, dlat, dalt])
    #     return dE, dN, dU


class WGS84(ReferenceEllipsoid):
    """Reference ellipsoid of the Earth, per the WGS84 specification."""

    def __init__(self):
        super().__init__(
            GM=co.STANDARD.WGS84.GM,
            a=co.STANDARD.WGS84.a,
            b=co.STANDARD.WGS84.b,
            omega=co.STANDARD.WGS84.omega
        )
        return