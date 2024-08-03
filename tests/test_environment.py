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
        h_sl = Quantity(0, "m")
        self.assertAlmostEqual(standard.a_n, static_model.speed_of_sound(h=h_sl), places=4)
        self.assertAlmostEqual(standard.l_n, static_model._mean_free_path(h=h_sl), places=4)
        self.assertAlmostEqual(standard.n_n, static_model._number_density(h=h_sl), delta=5e20)
        self.assertAlmostEqual(standard.vbar_n, static_model._mean_particle_speed(h=h_sl), places=2)
        self.assertAlmostEqual(standard.nu_n, static_model.kinematic_viscosity(h=h_sl), places=4)
        self.assertAlmostEqual(standard.lambda_n, static_model._thermal_conductivity(h=h_sl), places=4)
        self.assertAlmostEqual(standard.mu_n, static_model.dynamic_viscosity(h=h_sl), places=4)
        self.assertAlmostEqual(standard.omega_n, static_model._particle_collision_frequency(h=h_sl), delta=3e4)

        # Pressure at 80 km (assumed to be the same value as in the U.S. Standard Atmosphere tables
        self.assertAlmostEqual(Quantity(8.8627e-3, "mbar"), static_model.pressure(h=80e3), places=4)
        return


    def test_ussa_1976(self):
        static_model = ISO_2533_1975()

        # Pressure at 80 km
        self.assertAlmostEqual(Quantity(8.8627e-3, "mbar"), static_model.pressure(h=80e3), places=4)
        return

if __name__ == "__main__":
    unittest.main()
