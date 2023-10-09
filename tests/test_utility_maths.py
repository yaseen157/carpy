"""Tests for library maths methods."""
import unittest

import numpy as np

from carpy.utility import Quantity, interp_lin, interp_exp


class LinearInterpolation(unittest.TestCase):
    """Check the linear interpolator works okay."""

    def test_float(self):
        """Check the linear interpolator can handle a single scalar."""
        interp = interp_lin(0.2, [0, 1], [0, 100])
        self.assertEqual(20, interp, "Failed to interpolate scalar input")
        return

    def test_array(self):
        """Check the linear interpolator can handle a vector array."""
        interp = interp_lin([0.2, 0.3], [0, 1], [0, 100])
        self.assertTrue(all(np.array([20, 30]) == interp))
        return

    def test_quantity(self):
        """Check the linear interpolator can handle quantities."""
        interp = interp_lin([0.2, 0.3], [0, 1], Quantity([0, 100], "kg"))
        # Check value
        self.assertTrue(all(Quantity([20, 30], "kg") == interp))
        # Check type
        self.assertIsInstance(interp, Quantity)
        return


class ExponentialInterpolation(unittest.TestCase):

    def test_float(self):
        """Check the exponential interpolator can handle a single scalar."""
        interp = interp_exp(0.5, [0, 1], np.exp([0, 1]))
        self.assertTrue(np.isclose(np.exp(0.5), interp, atol=1e-5))
        return

    def test_array(self):
        """Check that the exponential interpolator can handle a vector array."""
        interp = interp_exp([0.5, 0.75], [0, 1], np.exp([0, 1]))
        self.assertTrue(all(np.isclose(np.exp([0.5, 0.75]), interp, atol=1e-5)))
        return

    def test_quantity(self):
        """Check that the exponential interpolator can handle quantities."""
        interp = interp_exp([0.5, 0.75], [0, 1], Quantity(np.exp([0, 1]), "kg"))
        # Check value
        self.assertTrue(
            all(np.isclose(np.exp([0.5, 0.75]), np.array(interp), atol=1e-5)))
        # Check type
        self.assertIsInstance(interp, Quantity)
        return


if __name__ == '__main__':
    unittest.main()
