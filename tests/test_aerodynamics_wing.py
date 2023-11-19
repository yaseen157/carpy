"""Test cases for aerodynamic methods relating to wing performance/geometry."""
import unittest

import numpy as np

from carpy.geometry import NewAerofoil, WingSections
from carpy.aerodynamics import (
    PrandtlLLT, HorseshoeVortex, MixedBLDrag, RaymerSimple)
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

    @classmethod
    def GudmundssonSR22(cls):
        """Return an approximation of S. Gudmundsson's Cirrus SR22 planform."""

        # Generate aerofoil geometries
        # This isn't really the Gudmundsson aerofoil selection either
        aerofoil = NewAerofoil.from_url(
            "http://airfoiltools.com/airfoil/lednicerdatfile"
            "?airfoil=naca652415-il")

        # Define buttock-line geometry of the wing
        mysections = WingSections(b=Quantity(38.3, "ft"))
        mysections[0] = aerofoil
        mysections[100] = aerofoil

        # Introduce wing taper
        mysections[0].chord = Quantity(4.88, "ft")
        mysections[100].chord = Quantity(2.59, "ft")

        return mysections


class Aerodynamics(unittest.TestCase):
    """Check methods relating to the generation of lift/drag results."""

    def test_prandtlllt(self):
        """Check that the prandtl LLT solver runs."""
        try:
            mysections = SampleWings.SUHPALazarus()
        except ConnectionError:
            self.skipTest(reason="Couldn't download aerofoil geometry")
            return

        aoa = np.radians(3)
        basekwargs = {"wingsections": mysections, "altitude": 0, "TAS": 10.0}
        soln = PrandtlLLT(**basekwargs, alpha=aoa)
        return

    def test_horseshoevortex(self):
        """Check that the horseshoe vortex solver runs."""
        try:
            mysections = SampleWings.SUHPALazarus()
        except ConnectionError:
            self.skipTest(reason="Couldn't download aerofoil geometry")
            return

        aoa = np.radians(3)
        basekwargs = {"wingsections": mysections, "altitude": 0, "TAS": 10.0}
        soln = HorseshoeVortex(**basekwargs, alpha=aoa)
        return

    def test_mixedbldrag(self):
        """
        Try to replicate the worked solution for skin friction drag.

        Notes:
            The mixed BL drag computed by this library can differ from
            Gudmundsson's result for two reasons:
            1) Gudmundsson prescribes upper and lower surface transition points
            for the aerofoil, which contribute to their own upper and lower
            surface drag components. This would be a function of angle of attack
            but this library opts to consider the average of the upper and lower
            transition, and assumes its 50% of the chord for all aerofoils.
            2) The wetted area is different. Not quite sure why Gudmundsson opts
            to consider the wetted area as being 7% greater than the planform
            area described in the book pages.

        """
        try:
            mysections = SampleWings.GudmundssonSR22()
        except ConnectionError:
            self.skipTest(reason="Couldn't download aerofoil geometry")
            return

        basekwargs = {"wingsections": mysections, "altitude": 0}
        soln = MixedBLDrag(**basekwargs, TAS=Quantity(185, "kt"))

        # Expected coefficient of skin friction
        self.assertTrue(np.isclose(soln.Cf, 0.001999, atol=1e-4))
        return

    def test_raymeroswald(self):
        """
        Compute a zeroth-order approximation of Oswald efficiency.
        """
        try:
            mysections = SampleWings.GudmundssonSR22()
        except ConnectionError:
            self.skipTest(reason="Couldn't download aerofoil geometry")
            return

        basekwargs = {"wingsections": mysections, "altitude": 0, "TAS": 0}
        soln = RaymerSimple(**basekwargs)

        self.assertAlmostEqual(soln.CD, 0.0030199, places=6)
        return


class Combinatorics(unittest.TestCase):
    """Test that interactions between Wing Solutions make sense"""

    def test_union(self):
        """Test if methods can combine their performance estimates."""
        try:
            mysections = SampleWings.SUHPALazarus()
        except ConnectionError:
            self.skipTest(reason="Couldn't download aerofoil geometry")
            return

        aoa = np.radians(3)
        basekwargs = {
            "wingsections": mysections, "altitude": 0, "TAS": 10.0, "alpha": aoa
        }
        soln0 = MixedBLDrag(**basekwargs)
        soln1 = PrandtlLLT(**basekwargs)

        union_solution = soln0 | soln1

        # Logical OR
        self.assertEqual(union_solution.CD, soln0.CD0 + soln1.CDi)

        return

    def test_addition(self):
        """Test if methods can combine their performance estimates."""
        try:
            mysections = SampleWings.SUHPALazarus()
        except ConnectionError:
            self.skipTest(reason="Couldn't download aerofoil geometry")
            return

        aoa = np.radians(3)
        basekwargs = {
            "wingsections": mysections, "altitude": 0, "TAS": 10.0, "alpha": aoa
        }
        soln0 = MixedBLDrag(**basekwargs)
        soln1 = PrandtlLLT(**basekwargs)

        addition_solution = soln0 + soln1

        # Addition
        self.assertEqual(addition_solution.CD, soln0.CD + soln1.CD)

        return
