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
        # Initial assignment
        stations = DiscreteIndex({1: 100, -2: -50})

        # Brand-new assignment
        stations[0] = 0

        # Update assignment
        stations[-2] = -200

        # Singular keys
        self.assertEqual(stations[-2], -200)
        self.assertEqual(stations[0], 0)
        self.assertEqual(stations[1], 100)

        # Real slices
        self.assertEqual(stations[0:], [0, 100])  # Start only
        self.assertEqual(stations[:0], [-200])  # Stop only
        self.assertEqual(stations[:0.5], [-200, 0])  # Stop only, new index
        self.assertEqual(stations[:], [-200, 0, 100])  # All defined indices

        # Interpolated keys
        self.assertEqual(stations[0.5], 50)

        # Interpolated slices
        self.assertEqual(stations[::4], [-200, -100, 0, 100])
        return


if __name__ == '__main__':
    unittest.main()
