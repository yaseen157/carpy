"""Test cases for aerodynamic methods relating to wing performance/geometry."""
import unittest

import numpy as np

from carpy.aerodynamics.aerofoil import NewAerofoil
from carpy.aerodynamics.wing import (
    WingSection, WingSections, PLLT, HorseshoeVortex)


class Solvers(unittest.TestCase):
    """Check that methods relating to the generation of lift/drag results."""

    @staticmethod
    def wing_SuperLazarusMkII():
        fx76 = NewAerofoil.from_url(
            "http://airfoiltools.com/airfoil/lednicerdatfile?airfoil=fx76mp140-il")
        dae31 = NewAerofoil.from_url(
            "http://airfoiltools.com/airfoil/lednicerdatfile?airfoil=dae31-il")

        # Define buttock-line geometry
        mysections = WingSections()
        mysections[0] = WingSection(fx76)
        mysections[6] = WingSection(fx76)
        mysections[10] = WingSection(fx76)
        mysections[14] = WingSection(dae31)

        # Add sweep, dihedral, and twist
        mysections[:6].sweep = np.radians(0)
        mysections[6:].sweep = np.radians(2)
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

        soln0 = PLLT(sections=mysections, span=24, alpha=np.radians(3))
        soln1 = HorseshoeVortex(sections=mysections, span=24,
                                alpha=np.radians(3))
        return
