"""
References:
    https://icgem.gfz-potsdam.de/
"""
import os
import re

from math import factorial
import numpy as np

from carpy.utility import PathAnchor, Quantity, is_none

__all__ = []
__author__ = "Yaseen Reza"

anchor = PathAnchor()


def spherical_harmonic_factory(GM, R, C, S):
    # Recast so as not to bother with units in the summing function
    R_ = float(R)

    def W_a(rad, lon, lat):
        """
        Compute the gravitational potential from spherical geocentric coordinates.

        Args:
            rad: Radius.
            lon: Longitude, in radians.
            lat: Latitude, in radians.

        Returns:
            Geopotential.

        """
        rad, lon, lat = np.broadcast_arrays(rad, lon, lat)
        geopot = np.zeros(rad.shape)
        max_degree = len(C) - 1

        # ===================================
        # Fully-normalised Legendre functions
        #   While at first glance it looks like you can replace it with SciPy's lpnm, it also must be normalised to
        #   account for the fact there are multiple conventions for Legendre results. While I did find the correct
        #   normalisation function through trial and error, it's hard to cite sources for such actions - so I went with
        #   this more rigorously documented approach (which theoretically performs better anyway as the normalised
        #   result is computed directly, and not subject to large exponentiation float precision errors).
        # https://mitgcm.org/~mlosch/geoidcookbook.pdf
        # Holmes and Featherstone (2002) use spherical coordinates, so we need to transform geocentric lat. to spherical
        t = np.sin(np.pi / 2 - lat)
        u = np.cos(np.pi / 2 - lat)

        # Compute fully normalised sectorial *seed* values
        P_mm = np.zeros((max_degree + 1,))
        P_mm[[0, 1]] = 1, (3 ** 0.5) * u
        for m in range(2, P_mm.size):
            P_mm[m] = u * ((2 * m + 1) / 2 / m) ** 0.5 * P_mm[m - 1]
        P_nm = np.identity(P_mm.size) * P_mm

        # Non-sectorials derive from seeds
        for ni in range(max_degree + 1):
            for mi in range(ni):  # Assures n > m
                common = (2 * ni + 1) / (ni - mi) / (ni + mi)
                a_nm = (common * (2 * ni - 1)) ** 0.5
                b_nm = (common * (ni + mi - 1) * (ni - mi - 1) / (2 * ni - 3)) ** 0.5

                # Although it looks sketchy when ni == 1 --> ni - 2 == -1, so it indexes back around and returns a zero
                P_nm[ni][mi] = a_nm * t * P_nm[ni - 1][mi] - b_nm * P_nm[ni - 2][mi]

        for degree_n in range(max_degree + 1):  # degree of spherical harmonic
            for order_m in range(degree_n + 1):  # order of spherical harmonic

                geopot = geopot + (
                        (R_ / rad) ** (degree_n + 1) *
                        (P_nm[degree_n][order_m] * (
                                C[degree_n][order_m] * np.cos(order_m * lon) +
                                S[degree_n][order_m] * np.sin(order_m * lon)
                        )
                         ))

        geopot = GM / R * geopot
        return geopot

    return W_a


def load_gfc(filename: str):
    filepath = os.path.join(anchor.directory_path, "data", filename)

    product_type = modelname = gravity_constant = radius = max_degree = errors = tide_system = None

    with open(filepath, "r") as f:

        # Assume metadata is contained within the first 20 lines
        for i in range(20):
            line = f.readline()

            if value := re.findall(r"product_type\s+(.+)", line):
                product_type = value[0]
            elif value := re.findall(r"modelname\s+(.+)", line):
                modelname = value[0]
            elif value := re.findall(r"gravity_constant\s+(.+)", line):
                gravity_constant = Quantity(float(value[0]), "m^3 s^-2")
            elif value := re.findall(r"radius\s+(.+)", line):
                radius = Quantity(float(value[0]), "m")
            elif value := re.findall(r"max_degree\s+(.+)", line):
                max_degree = int(value[0])
            elif value := re.findall(r"errors\s+(.+)", line):
                errors = value[0]
            elif value := re.findall(r"tide_system\s+(.+)", line):
                tide_system = value[0]
            elif line.startswith("end_of_head"):
                break

    if any(is_none(product_type, modelname, gravity_constant, radius, max_degree, errors)):
        error_msg = f"One or more meta-parameters of the model associated with '{filename}' could not determined"
        raise RuntimeError(error_msg)

    C, S = np.zeros((2, max_degree + 1, max_degree + 1))

    with open(filepath, "r") as f:

        for line in f:
            if not line.startswith("gfc"):
                continue  # Skip to the actual data

            # Unpack into matrices
            key, L, M, C_LM, S_LM = line.split()[0:5]
            L, M = map(int, (L, M))  # Turn string L and M into integer indices
            C[L][M] = C_LM
            S[L][M] = S_LM

    func = spherical_harmonic_factory(GM=gravity_constant, R=radius, C=C, S=S)
    print("Done")
    print(func(r := 6378e3, 0, lat := np.pi / 4), float(func(r, 0, lat) - func(r - 1, 0, lat)))
    print(gravity_constant / (Quantity(r, "m") ** 2))
    print()


class EGM96s:

    def __init__(self):
        filename = "EGM96s.gfc"
        load_gfc(filename=filename)
        return


egm = EGM96s()

# TODO: Figure out why the spherical harmonics aren't computing properly
#   It probably has something to do with needing to fully normalise the Stokes coefficients C and S
# raise NotImplementedError
