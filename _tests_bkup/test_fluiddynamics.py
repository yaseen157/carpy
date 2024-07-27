"""Tests for fluid dynamics related methods."""

import unittest

from carpy.fluiddynamics import Flow, FlowProcess
from carpy.fluiddynamics import Fluids


class FluidStates(unittest.TestCase):
    """Tests for fluids and stateful fluid objects."""

    def test_instantiation(self):
        """Confirm that built-in library fluids initialise properly."""

        # Each Cantera source is a composite phase object mechanism in Cantera
        cantera_sources = [x for x in dir(Fluids) if not x.startswith("_")]

        # For each Cantera source, find the contained fluid generation methods
        for src_name in cantera_sources:
            src_class = getattr(Fluids, src_name)
            fluid_names = [x for x in dir(src_class) if not x.startswith("_")]

            # For each fluid generation method, test if it instantiates
            for method_name in fluid_names:
                _ = getattr(src_class, method_name)()

        self.skipTest("Passing, no assertions made")
        return

    def test_mechanised_fluid(self):
        """Test the parameters of air, produced by the GRI30 class."""
        air = Fluids.GRI30.Air()

        # Test Cantera fluid instantiation parameters
        import cantera as ct
        self.assertIsInstance(air.fluid_object, ct.Solution)
        self.assertIsInstance(air.fluid_composition, str)
        self.assertEqual(air.fluid_mechanism, "gri30.yaml")

        # Test STP conditions (and __getattr__ inheritance)
        from carpy.utility import Quantity
        self.assertIsInstance(air.p, Quantity)
        self.assertIsInstance(air.T, Quantity)
        self.assertAlmostEqual(air.p, 101_325, places=0)
        self.assertAlmostEqual(air.T, 288.15, places=2)

        # Test own classes parameters
        self.assertIsInstance(air.gamma, float)
        self.assertAlmostEqual(air.gamma, 1.401, places=3)
        self.assertIsInstance(air.R, Quantity)
        self.assertAlmostEqual(air.R, 287.0, places=1)
        self.assertIsInstance(air.a, Quantity)
        self.assertAlmostEqual(air.a, 340.4, places=1)
        return

    def test_pure_fluid(self):
        """Test the parameters of hydrogen, from the built-in Cantera fluids."""
        air = Fluids.PureFluid.Hydrogen()

        # Test Cantera fluid instantiation parameters
        import cantera as ct
        self.assertIsInstance(air.fluid_object, ct.PureFluid)
        self.assertIsNone(air.fluid_composition)
        self.assertIsNone(air.fluid_mechanism)

        # Test STP conditions (and __getattr__ inheritance)
        from carpy.utility import Quantity
        self.assertIsInstance(air.p, Quantity)
        self.assertIsInstance(air.T, Quantity)
        self.assertAlmostEqual(air.p, 101_325, places=0)
        self.assertAlmostEqual(air.T, 288.15, places=2)

        # Test own classes parameters
        self.assertIsInstance(air.gamma, float)
        self.assertAlmostEqual(air.gamma, 1.381, places=3)
        self.assertIsInstance(air.R, Quantity)
        self.assertAlmostEqual(air.R, 4126.2, places=1)
        self.assertIsInstance(air.a, Quantity)
        self.assertAlmostEqual(air.a, 1281.7, places=1)
        return

    def test_perfect_monatomic(self):
        """Test ideal monatomic gases."""
        self.skipTest("Awaiting ideal gas back-end modelling for Perfect Gases")
        gas = Fluids.PerfectGas.Monatomic()

        # Test Cantera fluid instantiation parameters
        self.assertIsNone(gas.fluid_object)
        self.assertIsNone(gas.fluid_composition)
        self.assertIsNone(gas.fluid_mechanism)

        # Test STP conditions (and __getattr__ inheritance)
        from carpy.utility import Quantity
        self.assertIsInstance(gas.p, Quantity)
        self.assertIsInstance(gas.T, Quantity)
        self.assertAlmostEqual(gas.p, 101_325, places=0)
        self.assertAlmostEqual(gas.T, 288.15, places=2)

        # Test own classes parameters
        self.assertIsInstance(gas.gamma, float)
        self.assertAlmostEqual(gas.gamma, 1.667, places=3)
        return

    def test_perfect_diatomic(self):
        """Test ideal diatomic gases."""
        self.skipTest("Awaiting ideal gas back-end modelling for Perfect Gases")
        gas = Fluids.PerfectGas.Diatomic()

        # Test Cantera fluid instantiation parameters
        self.assertIsNone(gas.fluid_object)
        self.assertIsNone(gas.fluid_composition)
        self.assertIsNone(gas.fluid_mechanism)

        # Test STP conditions (and __getattr__ inheritance)
        from carpy.utility import Quantity
        self.assertIsInstance(gas.p, Quantity)
        self.assertIsInstance(gas.T, Quantity)
        self.assertAlmostEqual(gas.p, 101_325, places=0)
        self.assertAlmostEqual(gas.T, 288.15, places=2)

        # Test own classes parameters
        self.assertIsInstance(gas.gamma, float)
        self.assertAlmostEqual(gas.gamma, 1.400, places=3)
        return

    def test_perfect_triatomic(self):
        """Test ideal triatomic gases."""
        self.skipTest("Awaiting ideal gas back-end modelling for Perfect Gases")
        gas = Fluids.PerfectGas.Triatomic()

        # Test Cantera fluid instantiation parameters
        self.assertIsNone(gas.fluid_object)
        self.assertIsNone(gas.fluid_composition)
        self.assertIsNone(gas.fluid_mechanism)

        # Test STP conditions (and __getattr__ inheritance)
        from carpy.utility import Quantity
        self.assertIsInstance(gas.p, Quantity)
        self.assertIsInstance(gas.T, Quantity)
        self.assertAlmostEqual(gas.p, 101_325, places=0)
        self.assertAlmostEqual(gas.T, 288.15, places=2)

        # Test own classes parameters
        self.assertIsInstance(gas.gamma, float)
        self.assertAlmostEqual(gas.gamma, 1.333, places=3)
        return

    def test_perfect_trigonalplanar(self):
        """Test ideal monatomic gases."""
        self.skipTest("Awaiting ideal gas back-end modelling for Perfect Gases")
        gas = Fluids.PerfectGas.TrigonalPlanar()

        # Test Cantera fluid instantiation parameters
        self.assertIsNone(gas.fluid_object)
        self.assertIsNone(gas.fluid_composition)
        self.assertIsNone(gas.fluid_mechanism)

        # Test STP conditions (and __getattr__ inheritance)
        from carpy.utility import Quantity
        self.assertIsInstance(gas.p, Quantity)
        self.assertIsInstance(gas.T, Quantity)
        self.assertAlmostEqual(gas.p, 101_325, places=0)
        self.assertAlmostEqual(gas.T, 288.15, places=2)

        # Test own classes parameters
        self.assertIsInstance(gas.gamma, float)
        self.assertAlmostEqual(gas.gamma, 1.286, places=3)
        return


class FlowProcesses(unittest.TestCase):
    """Tests for thermodynamic processes on fluid objects (flow processes)."""

    def test_idealgas_wedgeshock(self):
        """Test the method that normal and oblique shocks depend upon."""
        # Create a Mach 5 airflow
        import numpy as np
        air = Fluids.Air.Air()
        airflow = Flow(air)
        airflow.M = 5

        # Check weak oblique shock
        process = FlowProcess.obliqueshock_w(airflow, geom_theta=np.radians(20))
        self.assertAlmostEqual(process.M2, 3.0182, places=4)
        self.assertAlmostEqual(process.geom_theta, np.radians(20), places=4)
        self.assertAlmostEqual(process.geom_beta, np.radians(29.8168), places=4)
        self.assertAlmostEqual(process.p2_p1, 7.04708, places=4)
        self.assertAlmostEqual(process.r2_r1, 3.31207, places=4)
        self.assertAlmostEqual(process.T2_T1, 2.12769, places=4)
        # The following fails:
        self.assertAlmostEqual(process.pt2_pt1, 0.50510, places=4)
        return


if __name__ == "__main__":
    unittest.main()
