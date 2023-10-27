"""Test cases for aerodynamic methods relating to wing performance/geometry."""
import unittest

import numpy as np

from carpy.aerodynamics.aerofoil import NewAerofoil
from carpy.aerodynamics.wing import (
    WingSection, WingSections, PrandtlLLT, HorseshoeVortex, MixedBLDrag)


class Solvers(unittest.TestCase):
    """Check that methods relating to the generation of lift/drag results."""

    @staticmethod
    def wing_SuperLazarusMkII():
        fx76 = NewAerofoil.from_url(
            "https://m-selig.ae.illinois.edu/ads/coord/fx76mp140.dat")
        dae31 = NewAerofoil.from_url(
            "https://m-selig.ae.illinois.edu/ads/coord/dae31.dat")

        # Define buttock-line geometry
        mysections = WingSections(b=26)
        mysections[0] = WingSection(fx76)
        mysections[6] = WingSection(fx76)
        mysections[10] = WingSection(fx76)
        mysections[14] = WingSection(dae31)

        # Add sweep, dihedral, and twist
        mysections[:6].sweep = np.radians(0)
        # mysections[6:].sweep = np.radians(2)
        mysections[6:].sweep = np.radians(0)
        mysections[0:].dihedral = np.radians(0)
        mysections[0:].twist = np.radians(0)
        mysections[14].twist = np.radians(-3)

        # Introduce wing taper
        mysections[0:].chord = 1.0
        mysections[14].chord = 0.3
        return mysections

    def test_running(self):
        """Check that the methods are even capable of running."""
        mysections = self.wing_SuperLazarusMkII()

        aoa = np.radians(3)
        basekwargs = {"wingsections": mysections, "altitude": 0, "TAS": 0}
        soln0 = MixedBLDrag(**basekwargs, alpha=aoa)
        soln1 = PrandtlLLT(**basekwargs, alpha=aoa)
        soln2 = HorseshoeVortex(**basekwargs, alpha=aoa)
        return

    def test_geometry(self):
        """Test that geometry methods work as intended."""

        n0012 = NewAerofoil.from_method.NACA("0012")

        mysections = WingSections(b=4)
        mysections[0] = WingSection(n0012)
        mysections[100] = WingSection(n0012)

        mysections[:].sweep = 0
        mysections[:].dihedral = 0
        mysections[:].twist = 0

        mysections[0].chord = 0.65
        mysections[100].chord = 0.24

        # Wing area is simply determined from trapezoidal area of the wing
        self.assertEqual(mysections.Sref, 1.78)

        # Wing mean geometric chord is simply the average of root and tip
        self.assertEqual(mysections.MGC, (0.65 + 0.24) / 2)
        self.assertEqual(mysections.SMC, (0.65 + 0.24) / 2)

        return


class GudmundssonSkinFriction(unittest.TestCase):

    def test_method(self):
        from carpy.utility import Quantity  # Idk why, but this only works here?

        n2412 = NewAerofoil.from_method.NACA("2412")

        mysections = WingSections(b=Quantity(38.3, "ft"))
        mysections[0] = WingSection(n2412)
        mysections[100] = WingSection(n2412)

        from carpy.utility import Quantity
        mysections[0].chord = Quantity(4.88, "ft")
        mysections[100].chord = Quantity(2.59, "ft")

        mysections[:].sweep = 0
        mysections[:].dihedral = 0
        mysections[:].twist = 0

        basekwargs = {"wingsections": mysections, "altitude": 0}
        soln = MixedBLDrag(**basekwargs, TAS=Quantity(185, "kt"))

        self.assertTrue(np.isclose(soln._Cf, 0.001999, atol=1e-4))

        return
