"""Module for creating and using gravitational field models."""
import os
import re
import typing

import numpy as np

from carpy.utility import PathAnchor, Quantity, classproperty

__all__ = ["GravFieldModel", "SHGravModel", "EGM96", "EGM96s", "JGM3", "WGS72"]
__author__ = "Yaseen Reza"

# Spawn a path anchor, so we know where we are in the system files
anchor = PathAnchor()
data_path = os.path.join(anchor.directory_path, "data")


class GravFieldModel:
    """
    Base class for modelling the attractive gravitational potential fields of celestial bodies.
    """
    _attraction_potential: typing.Callable
    _GM: Quantity
    _r_body: Quantity

    @property
    def GM(self) -> Quantity:
        """Standard gravitational parameter. Computed as the product of the gravitational constant G and body mass M."""
        return self._GM

    @property
    def r_body(self) -> Quantity:
        """Nominal radius of the body."""
        return self._r_body

    def attraction_potential(self, rad, lon, lat):
        """
        Compute and return the gravitational potential of a location, given the spherical planetocentric coordinates of
        the computation point.

        Args:
            rad: Spherical planetocentric radius.
            lon: Spherical planetocentric longitude.
            lat: Spherical planetocentric latitude.

        Returns:
            Gravitational potential.

        """
        rad = Quantity(rad, "m")
        rad, lon, lat = np.broadcast_arrays(rad, lon, lat, subok=True)
        return self._attraction_potential(rad=rad, lon=lon, lat=lat)


class SHGravModel(GravFieldModel):
    """Gravity model for celestial bodies, based on spherical harmonics."""
    _normCS: tuple[np.ndarray, np.ndarray]

    @property
    def normCS(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Fully normalised Stokes' coefficients.

        Returns:
            Tuple of two arrays corresponding to the normalised C and S coefficients, respectively.

        Notes:
            The shape of the arrays is ordered such that their values are indexed by {C,S}[degree][order]. Degree is
            typically given the symbol l or n, and the order is m.

        """
        return self._normCS

    @classproperty
    def models(cls) -> list[str]:
        """List the available gravity field models."""
        regex = re.compile(r"^(.+)\.gfc$")
        model_names = [x for sublist in map(regex.findall, os.listdir(data_path)) for x in sublist]
        return model_names

    def __init__(self, model_name: str):
        """
        Args:
            model_name: The name of a model to use. Valid models are listed in 'self.models'.
        """
        assert model_name in self.models, f"{model_name=} is not available. Available models include {self.models=}"

        # Open the file
        file_path = os.path.join(data_path, f"{model_name}.gfc")
        with open(file_path, "r") as f:

            # File reading preamble
            end_of_head = False
            regex_head = re.compile(r"^(\S+)\s+(\S+)$")
            column_required = {"L", "M", "C", "S"}
            column_names = []

            for line in f:
                line = line.strip()  # Remove whitespace and newline characters

                # Parse header information
                if end_of_head is False:

                    if match := regex_head.match(line):
                        key, value = match.groups()
                        if "gravity_constant" in key:
                            self._GM = Quantity(value, "m^3 s^-2")
                        elif "radius" in key:
                            self._r_body = Quantity(value, "m")
                        elif "end_of_head" in key:
                            end_of_head = True

                    elif not (column_required - set(temp := re.split(r"\s{2,}", line))):
                        column_names = temp

                    continue

                # Verify end of header information
                elif not column_names:
                    error_msg = f"Got '{end_of_head=}' but no column names were detected"
                    raise RuntimeError(error_msg)

                # Read body data
                row_data = line.split()
                if "data" not in locals():
                    data = dict(zip(column_names, [[x] for x in line.split()]))
                else:
                    [data[k].append(row_data[i]) for (i, k) in enumerate(column_names)]

        # POST PROCESSING
        n = np.array(data["L"], dtype=int)  # degree
        m = np.array(data["M"], dtype=int)  # order
        max_degree = n.max()
        C_nm, S_nm = np.zeros((2, max_degree + 1, max_degree + 1))  # Fully normalised Stokes' coefficients

        for i, (order, degree) in enumerate(zip(m, n)):
            C_nm[degree][order] = data["C"][i]
            S_nm[degree][order] = data["S"][i]
        else:
            self._normCS = C_nm, S_nm

        return

    def normLengendre(self, lat) -> np.ndarray:
        """
        Compute and return the fully-normalised Legendre function given the spherical coordinate system latitude.

        Args:
            lat: Spherical planetocentric latitude.

        Returns:
            Array of Legendre values for the given latitude.

        References:
            M. Losch and V. Seufer, “How to Compute Geoid Undulations (Geoid Height Relative to a Given Reference
                Ellipsoid) from Spherical Harmonic Coefficients for Satellite Altimetry Applications,” Dec. 2003.
                Accessed: Aug. 08, 2024. [Online]. Available: https://mitgcm.org/~mlosch/geoidcookbook.pdf
            S. A. Holmes and W. E. Featherstone, “A unified approach to the Clenshaw summation and the recursive
                computation of very high degree and order normalised associated Legendre functions,” Journal of Geodesy,
                vol. 76, no. 5, pp. 279–299, May 2002, doi: https://doi.org/10.1007/s00190-002-0216-2.

        """
        # Recast as necessary
        lat = np.atleast_1d(lat)
        # Fully-normalised Legendre functions
        #   While at first glance it looks like you can replace it with SciPy's lpnm, it also must be normalised to
        #   account for the fact there are multiple conventions for Legendre results. While I did find the correct
        #   normalisation function through trial and error, it's hard to cite sources for such actions - so I went with
        #   this more rigorously documented approach (which theoretically performs better anyway as the normalised
        #   result is computed directly, and not subject to large exponentiation float precision errors).
        # https://mitgcm.org/~mlosch/geoidcookbook.pdf

        # Holmes and Featherstone (2002) use the traditional definition of latitude in spherical space, so refactor.
        lat = np.pi / 2 - lat  # Lat=0 is no longer the body's equator, but now an argument from the pole
        ts = np.sin(lat)
        us = np.cos(lat)

        max_degree = max(self._normCS[0].shape) - 1
        out = np.zeros((lat.size, max_degree + 1, max_degree + 1))

        for i, lat in enumerate(lat):

            # Compute fully normalised sectorial *seed* values
            u = us.flat[i]
            P_mm = np.zeros((max_degree + 1,))
            P_mm[[0, 1]] = 1, (3 ** 0.5) * u
            for m in range(2, P_mm.size):
                P_mm[m] = u * ((2 * m + 1) / 2 / m) ** 0.5 * P_mm[m - 1]
            P_nm = np.identity(P_mm.size) * P_mm
            del P_mm

            # Non-sectorials derive from seeds
            t = ts.flat[i]
            for ni in range(max_degree + 1):
                for mi in range(ni):  # Assures n > m
                    common = (2 * ni + 1) / (ni - mi) / (ni + mi)
                    a_nm = (common * (2 * ni - 1)) ** 0.5
                    b_nm = (common * (ni + mi - 1) * (ni - mi - 1) / (2 * ni - 3)) ** 0.5

                    # Although when ni == 1 --> ni - 2 == -1, it indexes back around and returns a zero :)
                    P_nm[ni][mi] = a_nm * t * P_nm[ni - 1][mi] - b_nm * P_nm[ni - 2][mi]

            out[i] = P_nm

        return out.squeeze()

    def _attraction_potential(self, rad: Quantity, lon: np.ndarray, lat: np.ndarray):
        """
        Args:
            rad: Spherical planetocentric radius. Should be the same shape as lon and lat.
            lon: Spherical planetocentric longitude. Should be the same shape as rad and lat.
            lat: Spherical planetocnetric latitude. Should be the same shape as rad and lon.

        Returns:
            Gravitational potential.

        References:
            F. Barthelemez, “Definition of Functionals of the Geopotential and Their Calculation from Spherical
                Harmonic Models,” Jan. 2013, doi: https://doi.org/10.2312/GFZ.b103-0902-26.

        """
        gravitational_potential = np.zeros(rad.shape)
        R = self.r_body
        P_nm = self.normLengendre(lat=lat)
        C_nm, S_nm = self.normCS
        max_degree = max(C_nm.shape) - 1

        # Flatten all but last two dimensions
        P_nm = P_nm.reshape(P_nm.size // (max_degree + 1) ** 2, max_degree + 1, max_degree + 1)

        for degree_n in range(max_degree + 1):  # degree of spherical harmonic
            for order_m in range(degree_n + 1):  # order of spherical harmonic

                gravitational_potential = gravitational_potential + (
                        (R / rad.flatten()) ** (degree_n + 1) * (
                        P_nm[:, degree_n, order_m] * (
                        C_nm[degree_n][order_m] * np.cos(order_m * lon.flatten()) +
                        S_nm[degree_n][order_m] * np.sin(order_m * lon.flatten())
                )))

        gravitational_potential = self.GM / R * gravitational_potential
        return gravitational_potential


class EGM96(SHGravModel):
    """EGM96 Spherical Harmonic Gravity Field model."""

    def __init__(self):
        super(EGM96, self).__init__(model_name="EGM96")
        return


class EGM96s(SHGravModel):
    """EGM96s Spherical Harmonic Gravity Field model."""

    def __init__(self):
        super(EGM96s, self).__init__(model_name="EGM96s")
        return


class JGM3(SHGravModel):
    """JGM3 Spherical Harmonic Gravity Field model."""

    def __init__(self):
        super(JGM3, self).__init__(model_name="JGM3")
        return


class WGS72(SHGravModel):
    """WGS72 Spherical Harmonic Gravity Field model."""

    def __init__(self):
        super(WGS72, self).__init__(model_name="WGS72")
        return
