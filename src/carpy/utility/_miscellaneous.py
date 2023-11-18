"""A module of miscellaneous library utilities."""
import functools
import inspect
import os
import typing

import numpy as np
import pandas as pd
import yaml

__all__ = []
__author__ = "Yaseen Reza"


# ============================================================================ #
# Data types and casting
# ---------------------------------------------------------------------------- #
class Hint(object):
    """A static class of common argument typehints."""
    iter = typing.Union[tuple, list, np.ndarray]
    num = typing.Union[int, float, np.integer, np.inexact]
    nums = typing.Union[iter, num]
    func = typing.Union[typing.Callable]
    any = typing.Union[typing.Any]
    set = typing.Union[set, frozenset]


def cast2numpy(scalar_or_vector: Hint.any, /, dtype=None) -> np.ndarray:
    """
    Args:
        scalar_or_vector: Scalar or vector numerical argument to be recast.
        dtype: Datatype to cast as. Optional, defaults to None.

    Returns:
        The input argument recast as an array (if not already array like).

    Raises:
        ValueError: Couldn't determine a way to cast input as an array.

    """
    type_transforms = {
        (int, float, complex): lambda x: np.array([x]),
        (np.integer, np.inexact): lambda x: np.array([x]),
        (tuple, list, range): lambda x: np.array(x),
        (set, frozenset): lambda x: np.array([xi for xi in x]),
        (dict,): lambda x: {k: cast2numpy(v) for k, v in x.items()},
        (map, filter): lambda x: np.array(list(x)),
        (np.ndarray,): lambda x: x if x.shape != () else x[None],  # +1 dim.
        (pd.Series,): lambda x: np.array(x)
    }
    # Look for and try to apply a transform
    for types, transformer in type_transforms.items():

        if not isinstance(scalar_or_vector, types):
            continue  # Try a different transformer

        # No dtype specified, do not attempt to apply any casting rules
        if dtype is None:
            return transformer(scalar_or_vector)
        else:
            return transformer(scalar_or_vector).astype(dtype=dtype)

    # We tried all that we could
    errormsg = f"Couldn't cast '{type(scalar_or_vector)}' to numpy array"
    raise ValueError(errormsg)


def isNone(*args) -> tuple:
    """True or False for every arg that is None."""
    results = tuple(map(lambda x: x is None, args))
    return results if len(args) > 1 else results[0]


def collapse_array(scalar_or_vector):
    """
    Given a vector (or scalar), collapse any redundant nesting.

    Args:
        scalar_or_vector: An iterable to collapse.

    Returns:
        Lower dimensional representation of the input.

    """
    # If the "scalar_or_vector" is not iterable, simply return it
    if not isinstance(scalar_or_vector, Hint.iter.__args__):
        return scalar_or_vector

    # Else it is iterable. Do we lose anything for indexing?
    if len(scalar_or_vector) == 1:
        return collapse_array(scalar_or_vector[0])  # Recursively collapse

    # Yes, information is lost with further indexing. Return as is.
    else:
        return scalar_or_vector


__all__ += [Hint.__name__, cast2numpy.__name__, isNone.__name__,
            collapse_array.__name__]


# ============================================================================ #
# Pathing and files
# ---------------------------------------------------------------------------- #
class GetPath(object):
    """A static class of methods providing relevant file paths."""
    _library_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir))

    @classmethod
    def library(cls) -> str:
        """Return the absolute OS path of the library's installation."""
        return cls._library_path

    @staticmethod
    def localpackage(*optional_paths) -> str:
        """
        Return the absolute path of the caller file's local package.

        Args:
            *optional_paths: Optional extra paths to follow.

        Returns:
            Returns 'os.path.join(localpackage_path, *paths)'.

        """
        local_pkg_path = os.path.dirname(inspect.stack()[1].filename)
        extended_path = os.path.join(local_pkg_path, *optional_paths)
        return extended_path

    @staticmethod
    def github() -> str:
        """Return the URL to carpy's source."""
        return "https://github.com/yaseen157/carpy"


class LoadData(object):
    """A static class of methods to load various file types"""

    @staticmethod
    def yaml(filepath) -> dict:
        """Provided with a filepath, return a dictionary of discovered data."""
        with open(filepath, "r") as stream:
            try:
                dict2return = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return dict2return

    @staticmethod
    def excel(filepath, **kwargs) -> pd.DataFrame:
        """Provided with a filepath, return a DataFrame obj (kwargs apply)."""
        overwritable_defaults = {"sheet_name": None, "na_filter": False}
        updatedkwargs = {**overwritable_defaults, **kwargs}
        dataframes = pd.read_excel(io=filepath, **updatedkwargs)
        return dataframes


__all__ += [GetPath.__name__, LoadData.__name__]


# ============================================================================ #
# Decoration
# ---------------------------------------------------------------------------- #
def call_count(func):
    """Decorator to count the number of times a function is called."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Function wrapper that counts the number of function calls."""
        wrapper.call_count += 1
        return func(*args, **kwargs)

    wrapper.call_count = 0
    return wrapper


def call_depth(func):
    """
    Decorator to track the recursive depth of a function call.

    Args:
        func: Function to track the recursive entries of.

    Returns:
        Wrapped function.

    Examples:

        # Recursively multiply by two by as many times as desired
        >>> @call_depth
        ... def multiply_by_2(x, n_times=1):
        ...     # If maximum call depth is reached, ascend recursive stack
        ...     if multiply_by_2.call_depth == n_times:
        ...         return x
        ...     return 2 * multiply_by_2(x, n_times)

        >>> print(multiply_by_2(3, n_times=3))
        24

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Function wrapper that tracks the depth of function recursion."""
        wrapper.call_depth += 1
        result = func(*args, **kwargs)
        wrapper.call_depth -= 1
        return result

    wrapper.call_depth = -1
    return wrapper


__all__ += [call_count.__name__, call_depth.__name__]
