"""Tests for library utility functions."""
import unittest

import numpy as np

from carpy.utility import interp_lin, interp_exp
from carpy.utility import PathAnchor, Hint, cast2numpy
from carpy.utility import Quantity
from carpy.utility import Unicodify


class Maths(unittest.TestCase):
    """Tests for maths utilities."""

    def test_interp_lin_float(self):
        """Check the linear interpolator can handle a single scalar."""
        interp = interp_lin(0.2, [0, 1], [0, 100])
        self.assertEqual(20, interp, "Failed to interpolate scalar input")
        return

    def test_interp_lin_array(self):
        """Check the linear interpolator can handle a vector array."""
        interp = interp_lin([0.2, 0.3], [0, 1], [0, 100])
        self.assertTrue(all(np.array([20, 30]) == interp))
        return

    def test_interp_lin_quantity(self):
        """Check the linear interpolator can handle quantities."""
        interp = interp_lin([0.2, 0.3], [0, 1], Quantity([0, 100], "kg"))
        # Check value
        self.assertTrue(all(Quantity([20, 30], "kg") == interp))
        # Check type
        self.assertIsInstance(interp, Quantity)
        return

    def test_interp_exp_float(self):
        """Check the exponential interpolator can handle a single scalar."""
        interp = interp_exp(0.5, [0, 1], np.exp([0, 1]))
        self.assertTrue(np.isclose(np.exp(0.5), interp, atol=1e-5))
        return

    def test_interp_exp_array(self):
        """Check that the exponential interpolator can handle a vector array."""
        interp = interp_exp([0.5, 0.75], [0, 1], np.exp([0, 1]))
        self.assertTrue(all(np.isclose(np.exp([0.5, 0.75]), interp, atol=1e-5)))
        return

    def test_interp_exp_quantity(self):
        """Check that the exponential interpolator can handle quantities."""
        interp = interp_exp([0.5, 0.75], [0, 1], Quantity(np.exp([0, 1]), "kg"))
        # Check value
        self.assertTrue(
            all(np.isclose(np.exp([0.5, 0.75]), np.array(interp), atol=1e-5)))
        # Check type
        self.assertIsInstance(interp, Quantity)
        return


class Miscellaneous(unittest.TestCase):
    """Tests for miscellaneous utilities."""

    def test_hint(self):
        testcases = (
            (Hint.iter, (
                (1, 2, 3), [1, 2, 3], np.arange(3)
            )),
            (Hint.num, (
                1, 2.0, np.int32(5), np.float32(6.0)
            )),
            (Hint.nums, (
                (1, 2, 3), [1, 2, 3], np.arange(3),
                1, 2.0, np.int32(5), np.float32(6.0)
            )),
            (Hint.func, (
                lambda x: print(f"You'll never take me alive! (also {x})"),
                np.radians, print
            )),
            (Hint.set, (
                {1, 2, 3}, frozenset((1, 2, 3))
            ))
        )
        for hint, tests in testcases:
            # Unions of multiple types
            if type(hint) == "typing._UnionGenericAlias":
                for obj in tests:
                    self.assertTrue(
                        expr=isinstance(obj, hint.__args__),
                        msg=f"Couldn't assert {obj} is instance of {hint}"
                    )
            elif hasattr(hint, "__args__"):
                for obj in tests:
                    self.assertTrue(
                        expr=isinstance(obj, hint.__args__),
                        msg=f"Couldn't assert {obj} is instance of {hint}"
                    )
            else:
                for obj in tests:
                    self.assertTrue(
                        expr=isinstance(obj, hint),
                        msg=f"Couldn't assert {obj} is instance of {hint}"
                    )
        return

    def test_cast2numpy(self):
        # Hashables
        testcases = {
            -1: np.array([-1]), 0: np.array([0]), 1: np.array([1]),
            0.0: np.array([0.0]),
            1 + 2j: np.array([1 + 2j]),
            np.int32(3): np.array([3]),
            np.float32(4): np.array([4]),
            (2, 3): np.array([2, 3]),
            frozenset((2, 3)): np.array([2, 3]),
        }
        # Test hashables
        for testcase, goldresult in testcases.items():
            casting = cast2numpy(testcase)
            # Check the value came through alright
            self.assertTrue(
                expr=all(casting == goldresult),
                msg=f"Failed to cast {testcase} with correct value"
            )
            # Check that the datatype matches
            self.assertIsInstance(
                obj=casting,
                cls=type(goldresult),
                msg=f"Failed to cast {testcase} with correct type"
            )

        # Unhashables
        testcases = [
            ([2, 3], np.array([2, 3])), ({2, 3}, np.array([2, 3])),
            ({"a": 1, "b": {"c": 2}},
             {"a": np.array([1]), "b": {"c": np.array([2])}}),
            (np.array(-6), np.array([-6]))
        ]

        # Test unhashables
        for testcase, goldresult in testcases:
            casting = cast2numpy(testcase)
            # Check the value came through alright
            if not isinstance(testcase, dict):
                self.assertTrue(
                    expr=all(casting == goldresult),
                    msg=f"Failed to cast {testcase} with correct value"
                )
            else:
                self.assertTrue(
                    expr=(casting == goldresult),
                    msg=f"Failed to cast {testcase} with correct value"
                )
            # Check that the datatype matches
            self.assertIsInstance(
                obj=casting,
                cls=type(goldresult),
                msg=f"Failed to cast {testcase} with correct type"
            )

        return

    @staticmethod
    def test_pathing():
        anchor = PathAnchor()

        return


class UnitConversion(unittest.TestCase):
    """Tests for unit conversion utilities."""

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


class Vanity(unittest.TestCase):
    """Tests for 'vanity' methods."""

    def test_temperature(self):
        test_input = [(10., None), (10.0, None), (10.60, None), (10.606, None),
                      (10.606, "degF"), (10.32, "K"), (-3, "degR")]
        gold_output = ['+10°C', '+10°C', '+10.6°C', '+10.606°C',
                       '+10.606°F', '+10.32 K', '-3°R']
        for i, (temperature, unit) in enumerate(test_input):
            result = Unicodify.temperature(temperature, unit)
            self.assertEqual(first=result, second=gold_output[i])
        return

    def test_shallowscript(self):
        result = Unicodify.mathscript_safe("g_{0} = 9.81 ms^{2}")
        gold_output = "g₀ = 9.81 ms²"
        self.assertEqual(first=result, second=gold_output)
        return


if __name__ == "__main__":
    unittest.main()
