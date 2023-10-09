"""Tests for atmosphere models in carpy."""
import unittest

import numpy as np

from carpy.environment import ISA1975


class ISA1975Quantities(unittest.TestCase):
    """Test the ISA1975 atmosphere model."""

    def test_TPD(self):
        """Test temperature, pressure, and density functions."""
        isa = ISA1975()

        T0 = isa.T(altitude=0)
        p0 = isa.p(altitude=0)
        rho0 = isa.rho(altitude=0)

        # Test units
        self.assertTrue(T0.units.si_equivalent == "K")
        self.assertTrue(p0.units.si_equivalent == "kg m^{-1} s^{-2}")
        self.assertTrue(rho0.units.si_equivalent == "kg m^{-3}")

        # Test sea-level values
        self.assertTrue(T0 == 288.15)
        self.assertTrue(p0 == 101325)
        self.assertTrue(round(rho0, 3) == 1.225)

        # Test geopotential values
        self.assertTrue(round(isa.p(11e3)) == 22632)
        self.assertTrue(round(isa.p(20e3)) == 5475)
        self.assertTrue(round(isa.p(32e3)) == 868)
        self.assertTrue(round(isa.p(47e3)) == 111)
        self.assertTrue(round(isa.p(51e3)) == 67)
        self.assertTrue(round(isa.p(71e3)) == 4)

        # Test geometric values
        self.assertTrue(round(isa.p(11e3 + 19, geometric=True)) == 22632)
        self.assertTrue(round(isa.p(20e3 + 63, geometric=True)) == 5475)
        self.assertTrue(round(isa.p(32e3 + 162, geometric=True)) == 868)
        self.assertTrue(round(isa.p(47e3 + 350, geometric=True)) == 111)
        self.assertTrue(round(isa.p(51e3 + 413, geometric=True)) == 67)
        self.assertTrue(round(isa.p(71e3 + 802, geometric=True)) == 4)

        # Test that functions can handle arrays for pressure (geopotential)
        goldx_0 = np.array([11e3, 20e3, 32e3, 47e3, 51e3, 71e3])
        goldx_1 = np.array([11019, 20063, 32162, 47350, 51413, 71802])

        goldy_p = np.array([22632, 5475, 868, 111, 67, 4])
        self.assertTrue(np.isclose(isa.p(goldx_0).x, goldy_p, atol=1e-1).all())

        # Test that functions can handle arrays for pressure (geometric)
        self.assertTrue(np.isclose(
            isa.p(goldx_1, geometric=True).x,
            goldy_p,
            atol=1e-1
        ).all())

        print()

        return

    def test_airspeeds(self):
        """Test conversions between airspeed types."""
        isa = ISA1975()

        goldspeeds = {
            "CAS": np.array([124.20454408, 377.49844212]),
            "EAS": np.array([100., 200.]),
            "TAS": np.array([415.60659989, 831.21319978]),
            "Mach": np.array([1.40418125, 2.8083625])
        }

        for i, (k, v) in enumerate(goldspeeds.items()):
            # 70,000 [ft] == 21_336 [m]
            speeds = isa.airspeeds(altitude=21_336, geometric=False, **{k: v})
            cas, eas, tas, mach = speeds

            self.assertTrue(np.isclose(goldspeeds["CAS"], cas, atol=1e-1).all())
            self.assertTrue(np.isclose(goldspeeds["EAS"], eas, atol=1e-1).all())
            self.assertTrue(np.isclose(goldspeeds["TAS"], tas, atol=1e-1).all())
            self.assertTrue(
                np.isclose(goldspeeds["Mach"], mach, atol=1e-1).all())
