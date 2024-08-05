"""Tests for library utility functions."""
import unittest

import numpy as np

from carpy.utility import RationalNumber, gradient1d
from carpy.utility import PathAnchor
from carpy.utility import UnitOfMeasurement, Quantity
from carpy.utility import Unicodify


class Maths(unittest.TestCase):
    """Tests for maths utilities."""

    def test_rationalnums(self):
        num1 = RationalNumber(1, 3)
        num2 = RationalNumber(-8, 3)
        self.assertGreater(abs(num1), 0)
        self.assertGreater(abs(num2), 0)
        self.assertEqual(num1 + num2, -7 / 3)
        self.assertEqual(num1 + 2, 7 / 3)
        self.assertEqual(np.ceil(num1), 1)
        self.assertEqual(np.ceil(num2), -2)
        self.assertEqual(divmod(num2, num1), (-8, 0))
        self.assertEqual(num1, 1 / 3)
        self.assertEqual(num2, -8 / 3)
        self.assertEqual(num1 // 2, 0)
        self.assertEqual(num2 // 2, -2)
        self.assertGreaterEqual(num1, 0)
        self.assertGreaterEqual(num1, 1 / 3)
        self.assertGreater(num1, 0)
        self.assertEqual(int(num1), 0)
        self.assertEqual(int(num2), -2)
        self.assertLessEqual(num2, 0)
        self.assertLessEqual(num2, -8 / 3)
        self.assertLess(num2, 0)
        self.assertEqual(num1 * num2, -8 / 9)
        self.assertEqual(2 * num1, 2 / 3)
        self.assertEqual(-num1, -1 / 3)
        self.assertEqual(num2 ** 2, 64 / 9)
        self.assertAlmostEqual((num1 ** num1) ** 3, num1, places=5)
        self.assertEqual(num1 ** (num2 - num1), 27)
        self.assertEqual(num2 / num1, -8)

    def test_differentiate_scalar(self):
        # Numerical differentiation of an array
        x1 = np.linspace(-50, 50)
        y1 = x1 ** 3 + 3 * x1 - 4
        dydx_gold = 3 * x1 ** 2 + 3
        y_out, dydx_test = gradient1d(y1, x1, eps=1e-6)
        self.assertTrue(np.all(y1 == y_out))  # y_out is simply y1 that was passed in
        self.assertTrue(np.allclose(dydx_gold, dydx_test, rtol=1e-9))

        # Numerical differentiation of a function
        f1 = lambda x: x ** 3 + 3 * x - 4
        y_out, dydx_test = gradient1d(f1, x1, eps=1e-6)
        self.assertTrue(np.allclose(y1, y_out))  # y_out is an approximation of y1
        self.assertTrue(np.allclose(dydx_gold, dydx_test, rtol=1e-9))

        # Partial derivative of a function, using args
        f1 = lambda x, a: x ** 3 + 3 * x - a
        y_out, dydx_test = gradient1d(f1, x1, eps=1e-6, args=(4,))
        self.assertTrue(np.allclose(dydx_gold, dydx_test, rtol=1e-9))

        # Partial derivative of a function, using kwargs
        y_out, dydx_test = gradient1d(f1, x1, eps=1e-6, kwargs={"a": 4})
        self.assertTrue(np.allclose(dydx_gold, dydx_test, rtol=1e-9))

        return


class Miscellaneous(unittest.TestCase):
    """Tests for miscellaneous utilities."""

    @staticmethod
    def test_pathing():
        anchor = PathAnchor()

        return


class UnitConversion(unittest.TestCase):
    """Tests for unit conversion utilities."""

    def test_instantiation(self):
        """Verify Quantity can be instantiated with many arguments."""
        testcases = (1, (1,), [1], np.array(1), np.array([1]))

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
            # Mass
            ((1, "kg"), (1e3, "g")),
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
            ((1, "mm^{2}"), (1e-6, "m^{2}")),

        )
        # Test conversions
        for ((arg_val, arg_unit), (gold_val, tgt_unit)) in testcases:
            self.assertTrue(
                expr=np.isclose(
                    UnitOfMeasurement(arg_unit).to_si(arg_val),
                    UnitOfMeasurement(tgt_unit).to_si(gold_val)
                ),
                msg=(
                    f"Failed to convert {arg_val} {arg_unit} to "
                    f"{gold_val} {tgt_unit}"
                )
            )
        return

    def test_operations(self):
        """Methods to ensure that Quantity objects can do basic maths things."""

        mass = Quantity(85.4, "kg")
        velocity = Quantity([[-3.6], [0], [0]], "kph")  # 1 metre per second

        # Absolute value
        value = abs(velocity)
        self.assertEqual(value[0], 1)
        self.assertIsInstance(value[0], Quantity)

        # Addition
        value = mass + mass
        self.assertEqual(value, 170.8)
        self.assertIsInstance(value, Quantity)

        value = mass + 60
        self.assertEqual(value, 145.4)
        self.assertIsInstance(value, Quantity)

        # Ceiling
        value = np.ceil(mass)
        self.assertEqual(value, 86)
        self.assertIsInstance(value, Quantity)

        # Divmod
        value = divmod(mass, mass)
        self.assertEqual(value[0], 1)  # divides once
        self.assertEqual(value[1], 0)  # with no remainder
        self.assertIsInstance(value[0], Quantity)
        self.assertIsInstance(value[1], Quantity)

        value = divmod(mass, 5)
        self.assertEqual(value[0], 17)
        self.assertAlmostEqual(value[1], 0.4, places=6)
        self.assertIsInstance(value[0], Quantity)
        self.assertIsInstance(value[1], Quantity)

        # Equality
        self.assertTrue(mass == mass)
        self.assertTrue(mass == 85.4)
        self.assertFalse(mass == 2 * mass)
        self.assertFalse(np.any(mass == velocity))

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
