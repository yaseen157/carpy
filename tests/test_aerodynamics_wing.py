"""Test cases for aerodynamic methods relating to wing performance/geometry."""
import unittest

import numpy as np

from carpy.aerodynamics.aerofoil import NewAerofoil
from carpy.aerodynamics.wing import (
    WingSection, WingSections, PLLT, HorseshoeVortex, CDfGudmundsson
)  # , Cantilever1DStatic)


class Solvers(unittest.TestCase):
    """Check that methods relating to the generation of lift/drag results."""

    @staticmethod
    def wing_SuperLazarusMkII():
        fx76 = NewAerofoil.from_url(
            "https://m-selig.ae.illinois.edu/ads/coord/fx76mp140.dat")
        dae31 = NewAerofoil.from_url(
            "https://m-selig.ae.illinois.edu/ads/coord/dae31.dat")

        # Define buttock-line geometry
        mysections = WingSections()
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
        soln0 = PLLT(sections=mysections, span=24, alpha=aoa)
        soln1 = HorseshoeVortex(sections=mysections, span=24, alpha=aoa)
        # soln2 = Cantilever1DStatic(
        #     sections=mysections, spar=None, span=24, alpha=aoa,
        #     lift=115 * 9.81, N=60
        # )
        return

    def test_naca0012elevator(self):
        """Compare results against XFLR5 test case."""

        n0012 = NewAerofoil.from_method.NACA("0012")

        mysections = WingSections()
        mysections[0] = WingSection(n0012)
        mysections[100] = WingSection(n0012)

        mysections[:].sweep = 0
        mysections[:].dihedral = 0
        mysections[:].twist = 0

        mysections[0].chord = 0.65
        mysections[100].chord = 0.24

        aoa = np.radians(10)
        soln0 = PLLT(sections=mysections, span=4, alpha=aoa)
        soln1 = HorseshoeVortex(sections=mysections, span=4, alpha=aoa)

        return


class GudmundssonSkinFriction(unittest.TestCase):

    def test_method(self):
        n2412 = NewAerofoil.from_method.NACA("2412")

        mysections = WingSections()
        mysections[0] = WingSection(n2412)
        mysections[100] = WingSection(n2412)

        from carpy.utility import Quantity
        mysections[0].chord = Quantity(4.88, "ft")
        mysections[100].chord = Quantity(2.59, "ft")

        mysections[:].sweep = 0
        mysections[:].dihedral = 0
        mysections[:].twist = 0

        soln = CDfGudmundsson(
            sections=mysections, span=Quantity(38.3, "ft"),
            altitude=0, TAS=Quantity(185, "kt"))

        self.assertTrue(np.isclose(soln._Cf, 0.001999, atol=1e-4))

        return
