"""Tests for unit conversions methods."""
import unittest

import numpy as np

from carpy.utility import Quantity


class Quantities(unittest.TestCase):

    def test_instantiation(self):
        """Verify Quantity can be instantiated with many arguments."""
        testcases = (1, (1,), {1}, [1], np.array(1), np.array([1]))

        # No unit instantiation should just return an array, no questions asked
        for testcase in testcases:
            array_instance = Quantity(testcase)
            self.assertEqual(array_instance, 1)
            self.assertTrue(isinstance(array_instance, np.ndarray))
            self.assertTrue(isinstance(array_instance, Quantity))

        return

    def test_conversions(self):
        testcases = (
            # Temperature
            ((288.15, "K"), (288.15, "K")),
            ((288.15, "K"), (15.0, "degC")),
            ((288.15, "K"), (59.0, "degF")),
            # Power
            ((1, "nW"), (1e-9, "W")),
            ((1, "uW"), (1e-6, "W")),
            ((1, "mW"), (1e-3, "W")),
            ((1, "W"), (1, "W")),
            ((1, "kW"), (1e3, "W")),
            ((1, "MW"), (1e6, "W")),
            ((1, "GW"), (1e9, "W")),
            ((1, "hp"), (745.69987158, "W")),
            ((31, "dBm"), (1, "dBW")),
            ((0, "dBW"), (1, "W")),
            ((30, "dBm"), (1, "W")),
            # Pressure
            ((1, "bar"), (1e5, "Pa")),
            ((1, "bar"), (750.06157585, "mmHg")),
            ((1, "atm"), (29.92125558, "inHg")),
            # Homonuclear
            ((1, "mm^{2}"), (1e-6, "m^{2}"))

        )
        # Test conversions
        for ((arg_val, arg_unit), (gold_val, tgt_unit)) in testcases:
            self.assertTrue(
                expr=np.isclose(
                    Quantity(arg_val, arg_unit).to(tgt_unit),
                    gold_val
                ),
                msg=(
                    f"Failed to convert {arg_val} {arg_unit} to "
                    f"{gold_val} {tgt_unit}"
                )
            )
        return

    def test_operations(self):
        """Test mathematical and operational rigour of Quantity objects."""
        return


if __name__ == '__main__':
    unittest.main()
