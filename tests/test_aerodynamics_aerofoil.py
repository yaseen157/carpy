import unittest

from carpy.aerodynamics.aerofoil import NewAerofoil, ThinAerofoil


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
        return


class ThinAerofoilTheory(unittest.TestCase):
    """Methods to test thin aerofoil theory."""

    def test_liftcurveslope(self):
        """Thin aerofoil theory suggests ideal lift slope of 2 pi."""
        flatplate = NewAerofoil.from_method.NACA("0001")
        solution = ThinAerofoil(aerofoil=flatplate, alpha=0)
        Clalpha = solution.Clalpha
        self.assertAlmostEqual(Clalpha, 2 * 3.1415926535, places=5)
        return


if __name__ == '__main__':
    unittest.main()
