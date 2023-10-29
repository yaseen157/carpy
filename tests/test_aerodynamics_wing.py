"""Test cases for aerodynamic methods relating to wing performance/geometry."""
import unittest

import numpy as np

from carpy.aerodynamics.aerofoil import NewAerofoil
from carpy.aerodynamics.wing import (
    WingSections, PrandtlLLT, HorseshoeVortex, MixedBLDrag)
from carpy.utility import Quantity


class SampleWings(object):
    """Sample wings for other test methods to operate upon."""

    @classmethod
    def SUHPALazarus(cls):
        """Returns the planform for the SUHPA Lazarus (2021)."""

        # Generate aerofoil geometries
        fx76 = NewAerofoil.from_url(
            "https://m-selig.ae.illinois.edu/ads/coord/fx76mp140.dat")
        dae31 = NewAerofoil.from_url(
            "https://m-selig.ae.illinois.edu/ads/coord/dae31.dat")

        # Define buttock-line geometry of the wing
        mysections = WingSections(b=24)
        mysections[0] = fx76
        mysections[6] = fx76  # Brand-new object that has different parameters
        mysections[10] = fx76  # Brand-new object that has different parameters
        mysections[14] = dae31  # Completely different wing tip profile

        # Add sweep, dihedral, and twist
        mysections[6:].sweep = np.radians(2)  # Sweep stations 6, 10 (and 14)
        mysections[14].twist = np.radians(-3)

        # Introduce wing taper
        mysections[14].chord = 0.4

        return mysections


class Solvers(unittest.TestCase):
    """Check that methods relating to the generation of lift/drag results."""

    def test_running(self):
        """Check that the methods are even capable of running."""
        mysections = SampleWings.SUHPALazarus()

        aoa = np.radians(3)
        basekwargs = {"wingsections": mysections, "altitude": 0, "TAS": 10.0}
        soln0 = MixedBLDrag(**basekwargs, alpha=aoa)
        soln1 = PrandtlLLT(**basekwargs, alpha=aoa)
        soln2 = HorseshoeVortex(**basekwargs, alpha=aoa)
        return

    def test_geometry(self):
        """Test that geometry methods work as intended."""

        n0012 = NewAerofoil.from_method.NACA("0012")

        mysections = WingSections(b=4)
        mysections[0] = n0012
        mysections[100] = n0012

        mysections[0].chord = 0.65
        mysections[100].chord = 0.24

        # Wing area is simply determined from trapezoidal area of the wing
        self.assertEqual(mysections.Sref, 1.78)

        # Wing mean geometric chord is simply the average of root and tip
        self.assertEqual(mysections.MGC, (0.65 + 0.24) / 2)
        self.assertEqual(mysections.SMC, (0.65 + 0.24) / 2)

        return

    def test_mixedbldrag(self):
        """
        Try to replicate the worked solution for skin friction drag.

        One point of consideration RE: error in the output, Gudmundsson
        prescribes transition points for the upper and lower surfaces manually.
        The mixed boundary layer drag method of this library assumes that the
        change in position of upper and lower surface transition points can be
        modelled by an effectively constant "average" transition point
        (independent of angle of attack).

        """
        # This isn't really the Gudmundsson aerofoil selection either
        n2412 = NewAerofoil.from_method.NACA("2412")

        mysections = WingSections(b=Quantity(38.3, "ft"))
        mysections[0] = n2412
        mysections[100] = n2412

        mysections[0].chord = Quantity(4.88, "ft")
        mysections[100].chord = Quantity(2.59, "ft")

        basekwargs = {"wingsections": mysections, "altitude": 0}
        soln = MixedBLDrag(**basekwargs, TAS=Quantity(185, "kt"))

        # Expected coefficient of skin friction
        self.assertTrue(np.isclose(soln.Cf, 0.001999, atol=1e-4))

        return
