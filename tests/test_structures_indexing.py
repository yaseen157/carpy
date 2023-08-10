import unittest

from carpy.structures import ContinuousIndex, DiscreteIndex


class IndexTemplates(unittest.TestCase):
    """Test the template parent classes for cross-sectional stations."""

    def test_continuous_indexing(self):
        """Station data is defined with a continuous function."""
        stations = ContinuousIndex(lambda x: x * 2)
        self.assertEqual(stations[3], 6)
        return

    def test_discrete_indexing(self):
        """Station data is defined with discrete elements"""
        stations = DiscreteIndex({1: 100, -2: -200})
        stations[0] = 0
        self.assertEqual(stations[0:], (-0, 100))
        # Verify that station data can be linearly interpolated between ref.pts.
        self.assertEqual(stations[0.5], 50)
        return


if __name__ == '__main__':
    unittest.main()
