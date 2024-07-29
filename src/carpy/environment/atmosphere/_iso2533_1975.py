"""Module implementing the ISO 2533-1975 standard atmosphere."""
import warnings

import numpy as np
import pandas as pd

from carpy.environment.atmosphere._atmosphere import StaticAtmosphereModel
from carpy.utility import Quantity, constants as co

__all__ = ["ISO2533_1975"]
__author__ = "Yaseen Reza"

# def get_air_mix():
#     """
#     Dry, clean air composition at sea level.
#
#     Because Ozone (O3), Sulphur dioxide (SO2), Nitrogen dioxide (NO2) and
#     Iodine (I2) quantities can vary from time to time or place to place,
#     their contributions are omitted.
#     """
#     from carpy.chemistry import species, ChemicalMixture
#     composition = dict([
#         (species.nitrogen, 78.084), (species.oxygen, 20.947_6), (species.argon, 0.934),
#         (species.carbon_dioxide, 0.031_4),
#         (species.neon, 1.818e-3), (species.helium, 524.0e-6), (species.krypton, 114.0e-6),
#         (species.xenon, 8.7e-6), (species.hydrogen, 50.0e-6), (species.dinitrogen_oxide, 50.0e-6),
#         (species.methane, 0.2e-3)
#     ])
#     chemical_mixture = ChemicalMixture()
#     chemical_mixture.X = composition
#     return chemical_mixture


TABLES = dict()
TABLES[4] = pd.DataFrame(
    data={
        "H": Quantity([-2, 0, 11, 20, 32, 47, 51, 71, 80], "km"),
        "T": Quantity([301.15, 288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 196.65], "K")
    }
)


# Pre-compute pressure layers
def compute_pressure_bases():
    p_b = np.zeros(len(TABLES[4]))
    for i in range(p_b.size - 1):

        Tb = TABLES[4]["T"][i]
        dT = np.diff(TABLES[4]["T"][i:i + 2]).item()
        dH = np.diff(TABLES[4]["H"][i:i + 2]).item()
        beta = dT / dH

        # If temperature lapse rate is zero
        if beta == 0:
            multiplier = np.exp(-(co.STANDARD.ISO2533.g_n / co.STANDARD.ISO2533.R / (Tb + dT) * dH))
            p_b[i + 1] = p_b[i] * multiplier
            continue

        # If temperature lapse rate is non-zero
        multiplier = (1 + dT / Tb) ** -(co.STANDARD.ISO2533.g_n / beta / co.STANDARD.ISO2533.R)
        if i == 0:
            p_b[i] = co.STANDARD.ISO2533.p_n / multiplier
        p_b[i + 1] = p_b[i] * multiplier
    return p_b


p_b = compute_pressure_bases()


class ISO2533_1975(StaticAtmosphereModel):
    """ISO 2533-1975 Standard Atmosphere model."""

    def _pressure(self, h: Quantity) -> Quantity:
        # Broadcast h into a higher dimension
        h_broadcasted = np.broadcast_to(h, (*(TABLES[4]["H"]).shape, *h.shape))
        Href_broadcasted = np.expand_dims(TABLES[4]["H"], tuple(range(h_broadcasted.ndim - 1))).T

        i = np.sum(h_broadcasted > Href_broadcasted, axis=0) - 1
        i_lim = TABLES[4].index[-1]

        # Where the index limit has not been reached, find the base parameters of the current layer and the layer above
        Tb0 = TABLES[4]["T"].to_numpy()[i]
        Tb1 = np.where(i == i_lim, np.nan, TABLES[4]["T"].to_numpy()[np.clip(i + 1, None, i_lim)])
        Hb0 = TABLES[4]["H"].to_numpy()[i]
        Hb1 = np.where(i == i_lim, np.nan, TABLES[4]["H"].to_numpy()[np.clip(i + 1, None, i_lim)])
        beta = (Tb1 - Tb0) / (Hb1 - Hb0)

        # Intelligently determine the pressure depending on if the layer had a temperature lapse
        T = self.temperature(h=h)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # We're expecting divide by zero for the exponent with division by beta
            multiplier: np.ndarray = np.where(
                beta == 0,
                np.exp(-(co.STANDARD.ISO2533.g_n / co.STANDARD.ISO2533.R / T * (h - Hb0)).x),  # beta == 0
                (1 + beta / Tb0 * (h - Hb0)) ** -(co.STANDARD.ISO2533.g_n / beta / co.STANDARD.ISO2533.R).x  # beta != 0
            )
        p = p_b[i] * multiplier
        return Quantity(p, "Pa")

    def _temperature(self, h: Quantity) -> Quantity:
        T = np.interp(h, TABLES[4]["H"], TABLES[4]["T"])
        return Quantity(T, "K")

    def _density(self, h: Quantity) -> Quantity:
        Vm = self.molar_volume(h=h)
        rho = co.STANDARD.ISO2533.M / Vm
        return rho

    def _number_density(self, h: Quantity) -> Quantity:
        """
        Computes the number of neutral air particles per unit volume.

        Args:
            h: Geopotential altitude.

        Returns:
            The air number density.

        """
        p = self.pressure(h=h)
        T = self.temperature(h=h)
        n = co.STANDARD.ISO2533.N_A * p / co.STANDARD.ISO2533.Rstar / T
        return n

    def _mean_particle_speed(self, h: Quantity) -> Quantity:
        """
        Computes the mean air-particle speed.

        This speed is defined as the arithmetic average of air-particle speeds obtained from Maxwell's distribition of
        molecular speeds in the monatomic perfect gas under thermodynamical equilibrium conditions disregarding any
        exterior force.

        Args:
            h: Geopotential altitude.

        Returns:
            The mean air-particle speed.

        """
        T = self.temperature(h=h)
        vbar = (8 / np.pi * co.STANDARD.ISO2533.R * T) ** 0.5
        return vbar

    def _mean_free_path(self, h: Quantity) -> Quantity:
        """
        Computes the mean free path length of air particles.

        An air particle between two successive collisions moves uniformly along a straight line, passing a certain
        average distance called a mean free path of air particles.

        Args:
            h: Geopotential altitude.

        Returns:
            The mean free path of air particles.

        """
        n = self._number_density(h=h)
        l = 1 / (2 ** 0.5 * np.pi * co.STANDARD.ISO2533.sigma ** 2 * n)
        return l

    def _particle_collision_frequency(self, h: Quantity) -> Quantity:
        """
        Computes the air-particle collision frequency.

        The air-particle collision frequncy is the mean air-particle speed divided by the mean free path of air
        particles at the same altitude.

        Args:
            h: Geopotential altitude.

        Returns:
            The air-particle collision frequency.

        """
        vbar = self._mean_particle_speed(h=h)
        l = self._mean_free_path(h=h)
        omega = vbar / l
        return omega

    def _speed_of_sound(self, h: Quantity) -> Quantity:
        T = self.temperature(h=h)
        a = (co.STANDARD.ISO2533.kappa * co.STANDARD.ISO2533.R * T) ** 0.5
        return a

    def _dynamic_viscosity(self, h: Quantity) -> Quantity:
        T = self.temperature(h=h)
        mu = co.STANDARD.ISO2533.beta_S * T ** (3 / 2) / (T + co.STANDARD.ISO2533.S)
        return mu

    def _thermal_conductivity(self, h: Quantity) -> Quantity:
        T = self.temperature(h=h)
        lamda = 2.648_151e-3 * T ** (3 / 2) / (T + 245.4 * 10 ** -(12 / T))
        return lamda
