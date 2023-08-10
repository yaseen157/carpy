"""Tests for miscellaneous helper functions in carpy."""
import unittest

import numpy as np

from carpy.utility import (
    Hint,
    cast2numpy,
    GetPath
)


class HintsAndCasting(unittest.TestCase):

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


class Pathing(unittest.TestCase):

    @staticmethod
    def test_getpath():
        GetPath.library()
        GetPath.localpackage()
        return


if __name__ == '__main__':
    unittest.main()
