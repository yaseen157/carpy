import unittest

from carpy.aerodynamics.aerofoil import NewNDAerofoil


class Profiles(unittest.TestCase):
    """Methods to test aerofoil section generation and file parsing."""

    def test_generateNACA(self):
        # Four digit series
        n0012 = NewNDAerofoil.from_procedure.NACA("0012")
        n2412 = NewNDAerofoil.from_procedure.NACA("2412")
        n2412_63 = NewNDAerofoil.from_procedure.NACA("2412-63")
        # Five digit series
        n23012 = NewNDAerofoil.from_procedure.NACA("23012")
        n23012_45 = NewNDAerofoil.from_procedure.NACA("23012-45")
        n44112 = NewNDAerofoil.from_procedure.NACA("44112")
        # 16-series
        n16_012 = NewNDAerofoil.from_procedure.NACA("16-012")
        n16_912_3 = NewNDAerofoil.from_procedure.NACA("16-912,a=0.3")
        return


class ThinAerofoils(unittest.TestCase):
    """Methods to test thin aerofoil theory."""

    def test_liftcurveslope(self):
        flatplate = NewNDAerofoil.from_procedure.NACA("0001")
        Clalpha, = flatplate.Clalpha(0)
        self.assertAlmostEqual(Clalpha, 2 * 3.1415926535, places=5)
        return


if __name__ == '__main__':
    unittest.main()
