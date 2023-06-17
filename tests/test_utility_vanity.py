"""Tests for carpy methods designed to make things prettier."""
import unittest

from carpy.utility import Unicodify, Pretty


class MakePrettyMethods(unittest.TestCase):

    def test_pretty_temperature(self):
        test_input = [(10., None), (10.0, None), (10.60, None), (10.606, None),
                      (10.606, "degF"), (10.32, "K"), (-3, "degR")]
        gold_output = ['+10°C', '+10°C', '+10.6°C', '+10.606°C',
                       '+10.606°F', '+10.32 K', '-3°R']
        for i, (temperature, unit) in enumerate(test_input):
            result = Pretty.temperature(temperature, unit)
            self.assertEqual(first=result, second=gold_output[i])
        return


class UnicodeScripting(unittest.TestCase):

    def test_shallowscript(self):
        result = Unicodify.mathscript_safe("g_{0} = 9.81 ms^{2}")
        gold_output = "g₀ = 9.81 ms²"
        self.assertEqual(first=result, second=gold_output)
        return


if __name__ == '__main__':
    unittest.main()
