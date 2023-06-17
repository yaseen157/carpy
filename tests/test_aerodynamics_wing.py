"""Test cases for aerodynamic methods relating to wing performance/geometry."""
import unittest

import numpy as np

from carpy.aerodynamics.wing import Planforms
from carpy.utility import Quantity


class PlanformTesting(unittest.TestCase):

    # Note to future devs: Do not worry about the input values being exact.
    # The point is to check if the maths is being done correctly!

    def test_cirrus_sr22(self):
        """Test case for the Trapezoidal planform based on the Cirrus SR-22. """
        # Create wing object
        wing = Planforms.Trapezoidal(b=11.66, S=13.5)

        # Define shape
        wing.taper = 0.8
        wing.set_sweep(25, sweep=0)

        # Check: root and tip chord
        self.assertTrue(np.isclose(wing.cr, Quantity([1.2864494], units='m')))
        self.assertTrue(np.isclose(wing.ct, Quantity([1.02915952], units='m')))

        # Check: LE and TE sweep angles
        self.assertTrue(np.isclose(wing.sweepLE, 0.011032566078550767))
        self.assertTrue(np.isclose(wing.sweepTE, -0.03308696191673661))

        # Check: quarter chord sweep
        self.assertTrue(np.isclose(wing.sweepXX(25), 0))

        # Check: taper ratio
        self.assertEqual(wing.taper, 0.8)

        # Check: sigma position
        self.assertTrue(np.isclose(wing.sigma, 0.049999999999999975))

        return

    def test_suhpa_superlazarus(self):
        """Test case for Cranked planform based on the SUHPA Super Lazarus."""

        # Create wing object
        wing = Planforms.Cranked(b=24, S=21, yB=6)

        # Define shape
        wing.taper_io = 1, 0.4
        wing.set_sweep(25, sweep=0)

        # Check: span
        self.assertEqual(
            wing.b_io,
            (Quantity([12.], units='m'), Quantity([12.], units='m'))
        )

        # Check: chord at break
        self.assertTrue(np.isclose(wing.cB, Quantity([1.02941176], units='m')))

        # Check: chord at wing tip
        self.assertTrue(np.isclose(wing.ct, Quantity([0.4117647], units='m')))

        # Check: area
        self.assertTrue(np.isclose(
            wing.S_io,
            (Quantity([12.35294114], units='m^{2}'),
             Quantity([8.6470588], units='m^{2}'))
        ).all())

        # Check sweep
        self.assertTrue(np.isclose(
            wing.sweepLE_io, (0.0, 0.025729614758273463)).all())
        self.assertTrue(np.isclose(
            wing.sweepTE_io, (-0.0, -0.07705302682729681)).all())

        # Check: quarter chord sweep
        self.assertTrue(np.isclose(
            wing.sweepXX(25), (np.array([0.]), np.array([-3.46944695e-18]))
        ).all())

        # Check: taper
        self.assertEqual(wing.taper_io, (1.0, 0.4))

        # Check: kink point
        self.assertEqual(wing.yB, Quantity([6.], units='m'))
        self.assertEqual(wing.etaB, 0.5)
        return

# class GeometryTesting(unittest.Testcase):
#     pass
