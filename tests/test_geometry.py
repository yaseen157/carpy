"""Test geometry object generation and manipulation."""
import unittest

from carpy.geometry import ContinuousIndex, DiscreteIndex
from carpy.geometry import NewAerofoil


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


class Profiles(unittest.TestCase):
    """Methods to test aerofoil section generation and file parsing."""

    def test_generateNACA(self):
        # Four digit series
        n0012 = NewAerofoil.from_method.NACA("0012")
        n2412 = NewAerofoil.from_method.NACA("2412")
        n2412_63 = NewAerofoil.from_method.NACA("2412-63")
        # Five digit series
        n23012 = NewAerofoil.from_method.NACA("23012")
        n23012_45 = NewAerofoil.from_method.NACA("23012-45")
        n44112 = NewAerofoil.from_method.NACA("44112")
        # 16-series
        n16_012 = NewAerofoil.from_method.NACA("16-012")
        n16_912_3 = NewAerofoil.from_method.NACA("16-912,a=0.3")
        self.skipTest(reason="There is no test for NACA geometry generation.")
        return

    def test_readLednicer_online(self):
        """Check we can read Lednicer format online files correctly."""
        url = "http://airfoiltools.com/airfoil/lednicerdatfile?airfoil=n0012-il"
        try:
            n0012 = NewAerofoil.from_url(url)
        except ConnectionError:
            self.skipTest(reason="Couldn't download aerofoil geometry")
            return
        return

    def test_readSelig_online(self):
        """Check we can read Selig format online files correctly."""
        url = "http://airfoiltools.com/airfoil/seligdatfile?airfoil=n0012-il"
        try:
            n0012 = NewAerofoil.from_url(url)
        except ConnectionError:
            self.skipTest(reason="Couldn't download aerofoil geometry")
            return
        return


if __name__ == '__main__':
    unittest.main()
