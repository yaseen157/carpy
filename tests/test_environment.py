"""Tests for library environment methods."""
import unittest

import numpy as np

from carpy.environment.atmosphere import ISO2533_1975
from carpy.utility import Quantity, constants as co


class Atmosphere(unittest.TestCase):

    def test_iso2533_1975(self):
        static_model = ISO2533_1975()
        self.assertAlmostEqual(co.STANDARD.ISO2533.a_n, static_model.speed_of_sound(h=0), places=4)
        # self.assertAlmostEqual(co.STANDARD.ISO2533.l_n, static_model._mean_free_path())
        # self.assertAlmostEqual(co.STANDARD.ISO2533.n_n, static_model._number_density())
        # self.assertAlmostEqual(co.STANDARD.ISO2533.a_n, static_model._mean_particle_speed())
        self.assertAlmostEqual(co.STANDARD.ISO2533.nu_n, static_model.kinematic_viscosity(h=0), places=4)
        # self.assertAlmostEqual(co.STANDARD.ISO2533.lambda_n, static_model._thermal_conductivity())
        self.assertAlmostEqual(co.STANDARD.ISO2533.mu_n, static_model.dynamic_viscosity(h=0), places=4)
        # self.assertAlmostEqual(co.STANDARD.ISO2533.a_n, static_model._particle_collision_frequency())
        return


if __name__ == "__main__":
    unittest.main()