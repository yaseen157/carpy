"""Tests for atmosphere models in carpy."""
import unittest

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

        return
