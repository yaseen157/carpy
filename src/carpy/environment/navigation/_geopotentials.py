import os
import re
import typing

import numpy as np

from carpy.utility import PathAnchor, Quantity, is_none, classproperty

__all__ = []
__author__ = "Yaseen Reza"

# Spawn a path anchor, so we know where we are in the system files
anchor = PathAnchor()
data_path = os.path.join(anchor.directory_path, "data")


class GravFieldModel:
    """
    Base class for modelling the gravitational potential fields of celestial bodies.
    """
    _potential: typing.Callable
    _mu: Quantity
    _r_body: Quantity

    @property
    def mu(self) -> Quantity:
        """Standard gravitational parameter. Computed as the product of the gravitational constant G and body mass M."""
        return self._mu

    @property
    def r_body(self) -> Quantity:
        """Nominal radius of the body."""
        return self._r_body

    def potential(self, rad, lon, lat):
        """
        Compute and return the gravitational potential of a location, given the spherical planetocentric coordinates of
        the computation point.

        Args:
            rad: Spherical planetocentric radius.
            lon: Spherical planetocentric longitude.
            lat: Spherical planetocnetric latitude.

        Returns:
            Gravitational potential.

        """
        return self._potential(rad=rad, lon=lon, lat=lat)


class ICGEM(GravFieldModel):
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
                            self._mu = Quantity(value, "m^3 s^-2")
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

        """
        # Fully-normalised Legendre functions
        #   While at first glance it looks like you can replace it with SciPy's lpnm, it also must be normalised to
        #   account for the fact there are multiple conventions for Legendre results. While I did find the correct
        #   normalisation function through trial and error, it's hard to cite sources for such actions - so I went with
        #   this more rigorously documented approach (which theoretically performs better anyway as the normalised
        #   result is computed directly, and not subject to large exponentiation float precision errors).
        # https://mitgcm.org/~mlosch/geoidcookbook.pdf

        # Holmes and Featherstone (2002) use the traditional definition of latitude in spherical space, so refactor lat.
        lat = np.pi / 2 - lat  # Latitude of zero is no longer the celestial body's equator
        t = np.sin(lat)
        u = np.cos(lat)

        # Compute fully normalised sectorial *seed* values
        max_degree = max(self._normCS[0].shape) - 1
        P_mm = np.zeros((max_degree + 1,))
        P_mm[[0, 1]] = 1, (3 ** 0.5) * u
        for m in range(2, P_mm.size):
            P_mm[m] = u * ((2 * m + 1) / 2 / m) ** 0.5 * P_mm[m - 1]
        P_nm = np.identity(P_mm.size) * P_mm
        del P_mm

        # Non-sectorials derive from seeds
        for ni in range(max_degree + 1):
            for mi in range(ni):  # Assures n > m
                common = (2 * ni + 1) / (ni - mi) / (ni + mi)
                a_nm = (common * (2 * ni - 1)) ** 0.5
                b_nm = (common * (ni + mi - 1) * (ni - mi - 1) / (2 * ni - 3)) ** 0.5

                # Although it looks sketchy when ni == 1 --> ni - 2 == -1, so it indexes back around and returns a zero
                P_nm[ni][mi] = a_nm * t * P_nm[ni - 1][mi] - b_nm * P_nm[ni - 2][mi]

        return P_nm

    def _potential(self, rad, lon, lat):

        max_degree = max(self._normCS[0].shape) - 1
        rad = Quantity(rad, "m")
        geopot = np.zeros(rad.shape)
        R = self.r_body
        P_nm = self.normLengendre(lat=lat)
        C_nm, S_nm = self.normCS

        for degree_n in range(max_degree + 1):  # degree of spherical harmonic
            for order_m in range(degree_n + 1):  # order of spherical harmonic

                geopot = geopot + (
                        (R / rad) ** (degree_n + 1) *
                        (P_nm[degree_n][order_m] * (
                                C_nm[degree_n][order_m] * np.cos(order_m * lon) +
                                S_nm[degree_n][order_m] * np.sin(order_m * lon)
                        )
                         )
                )

        geopot = self.mu / R * geopot
        return geopot


if __name__ == "__main__":
    W1 = ICGEM("EGM96s")._potential(rad=6378e3, lon=0, lat=np.pi / 4)
    W2 = ICGEM("EGM96s")._potential(rad=6378e3 + 1, lon=0, lat=np.pi / 4)
    print(W2 - W1)
