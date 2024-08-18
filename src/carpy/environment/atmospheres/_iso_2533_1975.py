"""
Module implementing the ISO 2533-1975 standard atmosphere.

Notes:
    Where possible, this module draws the values for constants from the standard instead of computing values on the fly.

References:
    ISO 2533:1975 Standard Atmosphere.

"""
import warnings

import numpy as np
import pandas as pd

from carpy.physicalchem import species
from carpy.environment.atmospheres import StaticAtmosphereModel
from carpy.utility import Quantity, broadcast_vector, constants as co

__all__ = ["ISO_2533_1975", "ISA"]
__author__ = "Yaseen Reza"

# Dry, clean air composition at sea level.

# Because Ozone (O3), Sulphur dioxide (SO2), Nitrogen dioxide (NO2) and
# Iodine (I2) quantities can vary from time to time or place to place,
# their contributions are omitted.
air_composition = dict([
    (species.nitrogen(), 78.084), (species.oxygen(), 20.947_6), (species.argon(), 0.934),
    (species.carbon_dioxide(), 0.031_4), (species.neon(), 1.818e-3), (species.helium(), 524.0e-6),
    (species.krypton(), 114.0e-6), (species.xenon(), 8.7e-6), (species.hydrogen(), 50.0e-6),
    (species.dinitrogen_oxide(), 50.0e-6), (species.methane(), 0.2e-3)
])

TABLES = dict()

TABLES[2] = pd.DataFrame(
    data={
        "species": air_composition.keys(),
        "content": air_composition.values()
    }
)

TABLES[4] = pd.DataFrame(
    data={
        "H": Quantity([-2, 0, 11, 20, 32, 47, 51, 71, 80], "km"),
        "T": Quantity([301.15, 288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 196.65], "K")
    }
)


def geopotential_altitude(z):
    """
    Args:
        z: Geometric altitude, in metres.

    Returns:
        Geopotential altitude.

    """
    z = Quantity(z, "m")
    h = co.STANDARD.ISO_2533_1975.r * z / (co.STANDARD.ISO_2533_1975.r + z)
    return h


def compute_pressure_bases():
    pressure_bases = np.zeros(len(TABLES[4]))
    for i in range(pressure_bases.size - 1):

        Tb = TABLES[4]["T"][i]
        dT = np.diff(TABLES[4]["T"][i:i + 2]).item()
        dH = np.diff(TABLES[4]["H"][i:i + 2]).item()
        beta = dT / dH

        # If temperature lapse rate is zero
        if beta == 0:
            multiplier = np.exp(-(co.STANDARD.ISO_2533_1975.g_n / co.STANDARD.ISO_2533_1975.R / (Tb + dT) * dH))
            pressure_bases[i + 1] = pressure_bases[i] * multiplier
            continue

        # If temperature lapse rate is non-zero
        multiplier = (1 + dT / Tb) ** -(co.STANDARD.ISO_2533_1975.g_n / beta / co.STANDARD.ISO_2533_1975.R)
        if i == 0:
            pressure_bases[i] = co.STANDARD.ISO_2533_1975.p_n / multiplier
        pressure_bases[i + 1] = pressure_bases[i] * multiplier
    return pressure_bases


# Pre-compute pressure layers
p_b = compute_pressure_bases()


class ISO_2533_1975(StaticAtmosphereModel):
    """ISO 2533:1975 Standard Atmosphere model."""

    def __init__(self):
        super().__init__()

        # Define sea-level constituent gas composition
        self._fluid_model.X = dict(TABLES[2].to_records(index=False))
        return

    def __str__(self):
        rtn_str = f"ISO 2533:1975 Standard Atmosphere"
        return rtn_str

    @classmethod
    def _pressure(cls, z: Quantity) -> Quantity:
        h = geopotential_altitude(z=z)

        # Broadcast h into a higher dimension
        h_broadcasted, Href_broadcasted = broadcast_vector(values=h, vector=TABLES[4]["H"])

        i = np.clip(np.sum(h_broadcasted > Href_broadcasted, axis=0) - 1, 0, None)  # Prevent negative index
        i_lim = TABLES[4].index[-1]

        # Where the index limit has not been reached, find the base parameters of the current layer and the layer above
        Tb0 = TABLES[4]["T"].to_numpy()[i]
        Tb1 = np.where(i == i_lim, np.nan, TABLES[4]["T"].to_numpy()[np.clip(i + 1, None, i_lim)])
        Hb0 = TABLES[4]["H"].to_numpy()[i]
        Hb1 = np.where(i == i_lim, np.nan, TABLES[4]["H"].to_numpy()[np.clip(i + 1, None, i_lim)])
        beta = (Tb1 - Tb0) / (Hb1 - Hb0)

        # Intelligently determine the pressure depending on if the layer had a temperature lapse
        T = cls.temperature(z=z)
        beta[beta == 0] = np.nan  # Save ourselves the embarrassment of dividing by zero
        multiplier: np.ndarray = np.where(
            np.isnan(beta),
            np.exp(-(co.STANDARD.ISO_2533_1975.g_n / co.STANDARD.ISO_2533_1975.R / T * (h - Hb0)).x),  # beta == 0
            (1 + beta / Tb0 * (h - Hb0).x) ** -(co.STANDARD.ISO_2533_1975.g_n / beta / co.STANDARD.ISO_2533_1975.R).x
            # beta != 0
        )
        p = p_b[i] * multiplier
        return Quantity(p, "Pa")

    @classmethod
    def _temperature(cls, z: Quantity) -> Quantity:
        h = geopotential_altitude(z=z)
        T = np.interp(h, TABLES[4]["H"], TABLES[4]["T"], right=np.nan)
        if np.any(np.isnan(T)):
            warn_msg = f"The requested altitude(s) exceed the profile defined by this model (got h > 80 km)"
            warnings.warn(message=warn_msg, category=RuntimeWarning)
        return Quantity(T, "K")

    def _density(self, z: Quantity) -> Quantity:
        Vm = self.molar_volume(z=z)
        rho = co.STANDARD.ISO_2533_1975.M / Vm
        return rho

    def _number_density(self, z: Quantity) -> Quantity:
        """
        Computes the number of neutral air particles per unit volume.

        Args:
            z: Geometric altitude.

        Returns:
            The air number density.

        """
        p = self.pressure(z=z)
        T = self.temperature(z=z)
        n = co.STANDARD.ISO_2533_1975.N_A * p / co.STANDARD.ISO_2533_1975.Rstar / T
        return n

    def _mean_particle_speed(self, z: Quantity) -> Quantity:
        """
        Computes the mean air-particle speed.

        This speed is defined as the arithmetic average of air-particle speeds obtained from Maxwell's distribition of
        molecular speeds in the monatomic perfect gas under thermodynamical equilibrium conditions disregarding any
        exterior force.

        Args:
            z: Geometric altitude.

        Returns:
            The mean air-particle speed.

        """
        T = self.temperature(z=z)
        vbar = (8 / np.pi * co.STANDARD.ISO_2533_1975.R * T) ** 0.5
        return vbar

    def _mean_free_path(self, z: Quantity) -> Quantity:
        """
        Computes the mean free path length of air particles.

        An air particle between two successive collisions moves uniformly along a straight line, passing a certain
        average distance called a mean free path of air particles.

        Args:
            z: Geometric altitude.

        Returns:
            The mean free path of air particles.

        """
        n = self._number_density(z=z)
        l = 1 / (2 ** 0.5 * np.pi * co.STANDARD.ISO_2533_1975.sigma ** 2 * n)
        return l

    def _particle_collision_frequency(self, z: Quantity) -> Quantity:
        """
        Computes the air-particle collision frequency.

        The air-particle collision frequncy is the mean air-particle speed divided by the mean free path of air
        particles at the same altitude.

        Args:
            z: Geometric altitude.

        Returns:
            The air-particle collision frequency.

        """
        vbar = self._mean_particle_speed(z=z)
        l = self._mean_free_path(z=z)
        omega = vbar / l
        return omega

    def _speed_of_sound(self, z: Quantity) -> Quantity:
        T = self.temperature(z=z)
        a = (co.STANDARD.ISO_2533_1975.kappa * co.STANDARD.ISO_2533_1975.R * T) ** 0.5
        return a

    def _dynamic_viscosity(self, z: Quantity) -> Quantity:
        T = self.temperature(z=z)
        mu = co.STANDARD.ISO_2533_1975.beta_S * T ** (3 / 2) / (T + co.STANDARD.ISO_2533_1975.S)
        return mu

    def _thermal_conductivity(self, z: Quantity) -> Quantity:
        T = self.temperature(z=z).x  # Empirical formula does not demand consistency/propagation of units
        lambda_ = 2.648_151e-3 * T ** (3 / 2) / (T + 245.4 * 10 ** -(12 / T))
        return Quantity(lambda_, "W m^{-1} K^{-1}")


ISA = ISO_2533_1975
