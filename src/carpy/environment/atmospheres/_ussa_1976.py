"""
Module implementing the U.S. Standard Atmosphere as defined in 1976.

Notes:
    Where possible, this module draws the values for constants from the standard instead of computing values on the fly.

References:
    U.S. Standard Atmosphere 1976.

"""
import warnings

import numpy as np
import pandas as pd

from carpy.chemistry import species
from carpy.gaskinetics import PureGasModel
from carpy.environment.atmospheres import StaticAtmosphereModel
from carpy.utility import Quantity, broadcast_vector, constants as co

__all__ = ["USSA_1976"]
__author__ = "Yaseen Reza"

air_composition = dict([
    (species.nitrogen, .780_84), (species.oxygen, .209_476), (species.argon, .009_34),
    (species.carbon_dioxide, .000_314), (species.neon, .000_018_18), (species.helium, .000_005_24),
    (species.krypton, .000_001_14), (species.xenon, .000_000_087), (species.methane, .000_002),
    (species.hydrogen, .000_000_5)
])

TABLES = dict()

TABLES[3] = pd.DataFrame(
    data={
        "species": air_composition.keys(),
        "content": air_composition.values()  # Fractional volume, Fi
    }
)

TABLES[4] = pd.DataFrame(
    data={
        "H": Quantity([0, 11, 20, 32, 47, 51, 71, 84.8520], "km"),
        "L": Quantity([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, np.nan], "K km^{-1}")
    }
)

TABLES[5] = pd.DataFrame(
    data={
        "Z": Quantity([86, 91, 110, 120, 500, 1_000], "km"),
        "L": Quantity([0.0, np.nan, 12.0, np.nan, np.nan, np.nan], "K km^{-1}")
    },
    index=range(7, 13)  # indices 7 through 12 inclusive
)

TABLES[8] = pd.DataFrame(
    data={
        "H": Quantity(list(range(79_000, 85_000, 500)) + [84_852], "m"),
        "M_M0": .999 + 1e-6 * np.array([1000, 996, 988, 969, 938, 904, 864, 822, 778, 731, 681, 679, 579])
    }
)

TABLES[9] = pd.DataFrame(
    data={
        "species": ["N2", "O", "O2", "Ar", "He"],
        # The helium number density should be to the power of 14 at 86 km according to my calcs, Table 26 backs this up
        "n": [1.129_793e20, 8.6e16, 3.030_898e19, 1.351_400e18, 7.581_7e10]
    }
)


def geometric_altitude(h):
    """
    Args:
        h: Geopotential altitude, in metres.

    Returns:
        Geometric altitude.

    """
    h = Quantity(h, "m")

    selection = (denominator := co.STANDARD.USSA_1976.r_0 - h) <= 0
    if np.any(selection):
        denominator[selection] = np.nan
        warn_msg = f"Invalid values for geopotential altitude encountered (h > radius of Earth)"
        warnings.warn(message=warn_msg, category=RuntimeWarning)

    z = co.STANDARD.USSA_1976.r_0 * h / denominator

    return z


def molecular_weight_ratio(h):
    """
    Computes the ratio of molecular weight of atmospheric air at altitude, to that of sea level. Typically, M / M0 <= 1.

    Args:
        h: Geopotential altitude, in metres.

    Returns:
        Ratio of molecular weight to that defined at standard sea-level.

    """
    M = np.interp(h, *zip(*TABLES[8].to_records(index=False)))
    return M


# Shortly after equation 21, a molar mass for sea-level air is given. We use this as the preferred molar mass for
#   computations as it produces results that more closely match the data tables in the standard than those using molar
#   mass computed by this library's gas models. This seems in part due to differences in the values used for atomic mass
molar_masses = [28.0134, 31.9988, 39.948, 44.00995, 20.183, 4.0026, 83.8, 131.3, 16.04303, 2.01594]
M_0 = Quantity(
    sum([molar_masses[i] * x for i, x in enumerate(air_composition.values())]),
    "kg kmol^{-1}"
)


def compute_bases_molecular():
    # Initialise arrays, populate first element with sea-level conditions
    layers_model1 = len(TABLES[4])
    temperature_bases, pressure_bases = np.zeros((2, layers_model1))
    temperature_bases[0] = co.STANDARD.USSA_1976.T_0
    pressure_bases[0] = co.STANDARD.USSA_1976.P_0

    # For model 1, 0 km to 84.852 km geopotential (86 km geometric) (molecular scale)
    # Layers 0 to 7 (inclusive)
    for i in range(layers_model1 - 1):
        # Compute the temperature in the layer above
        dH = np.diff(TABLES[4]["H"][i:i + 2]).item()
        Lm = TABLES[4]["L"][i]
        temperature_bases[i + 1] = temperature_bases[i] + dH * Lm

        # Compute the pressure in the layer above
        Rspecific = co.STANDARD.USSA_1976.Rstar / M_0
        if Lm != 0:  # Equation (33a)
            multiplier = temperature_bases[i] / (temperature_bases[i] + Lm * dH)
            multiplier **= co.STANDARD.USSA_1976.g_0 / Rspecific / Lm
        else:  # Equation (33b)
            multiplier = np.exp(-co.STANDARD.USSA_1976.g_0 * dH / Rspecific / temperature_bases[i])
        pressure_bases[i + 1] = pressure_bases[i] * multiplier

    return temperature_bases, pressure_bases


Tm_b, P_b = compute_bases_molecular()


def compute_Tbases_kinetic():
    # Model 1
    # For model 2, 86 km geometric (84.852 km geopotential) to 1,000 km geometric (kinetic scale)
    # Layer 7 base initialise
    # The molecular scale results seed the kinetic scale temperature layers ...
    #   ... however, the concept of a molecular scale temperature no longer applies. Even though the end of table
    #   4 and start of table 5 are the same altitude, we actually now care about the kinetic temperature being recorded
    T_7 = Tm_b[-1] * molecular_weight_ratio(h=84_852)

    # Layer 8 base is just layer 7-8 is isothermal
    T_8 = T_7

    # Layer 9 base
    dZ = np.diff(TABLES[5]["Z"][[8, 9]]).item()
    Tc = 263.1905
    A = -76.3232
    a = Quantity(-19.9429, "km")
    T_9 = np.round(Tc + A * (1 - (dZ / a) ** 2) ** 0.5, 3).item()  # three decimal places

    # Layer 10 base
    dZ = np.diff(TABLES[5]["Z"][[9, 10]]).item()
    Lk_9 = TABLES[5]["L"][9]
    T_10 = T_9 + Lk_9 * dZ

    # Layer 11 isn't really a thing in the maths, the function from layer 10 base onwards goes all the way to layer 12
    T_11 = np.nan

    # Layer 12 base: 1,000 km geometric altitude
    Z_10, Z_12 = TABLES[5]["Z"][[10, 12]]
    lamda = Lk_9 / (co.STANDARD.USSA_1976.T_inf.x - T_10)
    xi = (Z_12 - Z_10) * ((co.STANDARD.USSA_1976.r_0 + Z_10) / (co.STANDARD.USSA_1976.r_0 + Z_12))
    T_12 = (co.STANDARD.USSA_1976.T_inf - (co.STANDARD.USSA_1976.T_inf - T_10) * np.exp(- lamda * xi)).item()

    temperature_bases = np.array([T_7, T_8, T_9, T_10, T_11, T_12])
    return temperature_bases


# Pre-compute temperature layers
T_b = compute_Tbases_kinetic()


def compute_number_densities():
    T_7 = compute_Tbases_kinetic()[0]
    return


class USSA_1976(StaticAtmosphereModel):
    """U.S. Standard Atmosphere model (1976)."""

    def __init__(self):
        super().__init__()

        # Define sea-level constituent gas composition
        self._gas_model.X = {
            PureGasModel(chemical_species=chemical_species): content_fraction
            for (chemical_species, content_fraction) in dict(TABLES[3].to_records(index=False)).items()
        }

        # Define planet
        self._planet = "Earth"
        return

    def __str__(self):
        rtn_str = f"U.S. Standard Atmosphere 1976"
        return rtn_str

    def _pressure(self, h: Quantity) -> Quantity:
        # Broadcast h into a higher dimension
        h_broadcasted, Href_broadcasted = broadcast_vector(values=h, vector=TABLES[4]["H"])

        i = np.clip(np.sum(h_broadcasted > Href_broadcasted, axis=0) - 1, 0, None)  # Prevent negative index
        i_lim = TABLES[4].index[-1]

        # Where the index limit has not been reached, find the base parameters of the current layer and the layer above
        Hb = TABLES[4]["H"].to_numpy()[i]
        Lm = np.where(i == i_lim, np.nan, TABLES[4]["L"].to_numpy()[i])
        Rspecific = co.STANDARD.USSA_1976.Rstar / M_0

        Lm[Lm == 0] = np.nan  # Save ourselves the embarrassment of dividing by zero
        multiplier: np.ndarray = np.where(
            np.isnan(Lm),
            np.exp(-(co.STANDARD.USSA_1976.g_0 / Rspecific / Tm_b[i] * (h - Hb)).x),  # beta == 0
            (1 + Lm / Tm_b[i] * (h - Hb).x) ** -(co.STANDARD.ISO_2533_1975.g_n / Lm / co.STANDARD.ISO_2533_1975.R).x
            # beta != 0
        )
        p = P_b[i] * multiplier

        select_threshold = h > 84_852
        if np.any(select_threshold):
            p[select_threshold] = np.nan
            warn_msg = f"The requested altitude(s) exceed the profile currently implemented (got h > 84,852 metres)"
            warnings.warn(message=warn_msg, category=RuntimeWarning)

        return Quantity(p, "Pa")

    def _temperature(self, h: Quantity) -> Quantity:
        # Broadcast h into a higher dimension
        h_broadcasted, Href_broadcasted = broadcast_vector(h, TABLES[4]["H"])
        # Convert to geometric altitude z and also broadcast
        z = geometric_altitude(h=h)
        z_broadcasted, Zref_broadcasted = broadcast_vector(z, TABLES[5]["Z"])

        # Selection indices for molecular and kinetic scales. Value of 0 or greater means that indexing system is active
        idxm = np.clip(np.sum(h_broadcasted > Href_broadcasted, axis=0) - 1, 0, None)  # Prevent negative index of tab 4
        idxk = np.sum(z_broadcasted > Zref_broadcasted, axis=0) - 1  # Index of table 5 if it used default indices
        idxk[np.isnan(z)] = len(TABLES[5]) - 1  # If z was invalid, set the layer to end index
        # Combine into an index that is consistent with the layer numbers in the standard
        idxm_limit = TABLES[4].index[-1]
        layer = np.where(idxk < 0, idxm, idxk + idxm_limit)  # Table 4 and 5 combined index

        # Compute the molecular temperature where it is applicable in the geopotential regions
        dH = h.x - np.where(layer < idxm_limit, TABLES[4]["H"].to_numpy()[np.clip(idxm, 0, None)], np.nan)
        Tm = np.interp(idxm, range(Tm_b.size), Tm_b) + dH * TABLES[4]["L"].to_numpy()[idxm]
        # Convert to the kinetic temperature we care about
        T = Tm * molecular_weight_ratio(h=h)

        # Anything left that is not a number is in the kinetic region
        dZ = z.x - np.where(layer >= idxm_limit, TABLES[5]["Z"].to_numpy()[idxk], np.nan)

        # Layer 7 is isothermal
        T_7, T_8, T_9, T_10, T_11, T_12 = T_b
        T[layer == 7] = T_7

        # Layer 8 is elliptical
        Tc = 263.1905
        A = -76.3232
        a = Quantity(-19.9429, "km")
        T[layer == 8] = Tc + A * (1 - (dZ[layer == 8] / a) ** 2) ** 0.5

        # Layer 9 is linear
        Lk_9 = TABLES[5]["L"][9]
        T[layer == 9] = T_9 + Lk_9 * dZ[layer == 9]

        # Layer 10 and 11 are not distinct mathematically
        Z_10 = TABLES[5]["Z"][10]
        lamda = Lk_9 / (co.STANDARD.USSA_1976.T_inf.x - T_10)
        xi = ((z.x - Z_10) * ((co.STANDARD.USSA_1976.r_0 + Z_10) / (co.STANDARD.USSA_1976.r_0 + z.x)))[layer >= 10]
        T[layer >= 10] = (co.STANDARD.USSA_1976.T_inf - (co.STANDARD.USSA_1976.T_inf - T_10) * np.exp(- lamda * xi))

        # Layer 12 is defined at its base and no further
        if np.any(selection := z.x > max(TABLES[5]["Z"])):
            T[selection] = np.nan
            warn_msg = f"The requested altitude(s) exceed the profile defined by this model (got z > 1,000 km)"
            warnings.warn(message=warn_msg, category=RuntimeWarning)

        return Quantity(T, "K")

    def _number_density(self, h: Quantity) -> Quantity:
        """
        Computes the number of neutral air particles per unit volume.

        Args:
            h: Geopotential altitude.

        Returns:
            The air number density.

        """
        # Default to the sea-level number density
        N_0 = co.STANDARD.USSA_1976.P_0 / (co.STANDARD.USSA_1976.k * co.STANDARD.USSA_1976.T_0)
        N = np.ones(h.shape) * N_0

        # Adjustment by the molecular weight ratio
        M_M0 = molecular_weight_ratio(h=h)
        N *= (1 / M_M0)  # As the molecular weight ratio decreases, N increases

        z = geometric_altitude(h=h)
        if np.any(z > 86e3):
            error_msg = "Cannot yet compute total number density above 86 km geometric altitudes"
            raise NotImplementedError(error_msg)

        return N
