"""Tests for library environment methods."""
import unittest

import numpy as np

from carpy.environment.atmospheres import ISO_2533_1975, USSA_1976
from carpy.utility import Quantity, constants as co


class Atmosphere(unittest.TestCase):

    def test_iso_2533_1975(self):
        static_model = ISO_2533_1975()

        # Start with sea-level tests
        standard = co.STANDARD.ISO_2533_1975
        z_sl = Quantity(0, "m")
        self.assertAlmostEqual(standard.a_n, static_model.speed_of_sound(z=z_sl), places=4)
        self.assertAlmostEqual(standard.l_n, static_model._mean_free_path(z=z_sl), places=4)
        self.assertAlmostEqual(standard.n_n, static_model._number_density(z=z_sl), delta=5e20)
        self.assertAlmostEqual(standard.vbar_n, static_model._mean_particle_speed(z=z_sl), places=2)
        self.assertAlmostEqual(standard.nu_n, static_model.kinematic_viscosity(z=z_sl), places=4)
        self.assertAlmostEqual(standard.lambda_n, static_model._thermal_conductivity(z=z_sl), places=4)
        self.assertAlmostEqual(standard.mu_n, static_model.dynamic_viscosity(z=z_sl), places=4)
        self.assertAlmostEqual(standard.omega_n, static_model._particle_collision_frequency(z=z_sl), delta=3e4)

        # Pressure at 80 km (assumed to be the same value as in the U.S. Standard Atmosphere tables
        self.assertAlmostEqual(Quantity(1.0524e-2, "mbar"), static_model.pressure(z=80e3), places=3)
        return

    def test_ussa_1976(self):
        static_model = USSA_1976()

        # Pressure at 80km
        self.assertAlmostEqual(Quantity(1.0524e-2, "mbar"), static_model.pressure(z=80e3), places=3)
        return


if __name__ == "__main__":
    unittest.main()
