"""A module of miscellaneous library utilities."""
import functools
import inspect
import os
import re
import typing
from warnings import warn

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
    int = typing.Union[int, np.integer]
    real = typing.Union[float, np.inexact]
    iter = typing.Union[tuple, list, np.ndarray]
    num = typing.Union[int, real]
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


class NumberSets:

    @staticmethod
    def is_C(value, /) -> bool:
        """True if value is of a Python type that allows complex numbers."""
        return np.iscomplex(value)

    @staticmethod
    def is_R(value, /) -> bool:
        """True if value is of a Python type that allows real numbers."""
        return np.isfinite(value)

    @staticmethod
    def is_Z(value, /) -> bool:
        """True if value is of a Python type that allows integers."""
        return isinstance(value, (int, np.integer))

    @classmethod
    def cast_R(cls, value, /) -> float:
        """Return real number."""
        if cls.is_R(value):
            return value
        raise ValueError(f"Couldn't cast '{value}' to float (real number set)")

    @classmethod
    def cast_Z(cls, value, /, *, safe: bool = False) -> int:
        """
        Return integer number.

        Raises:
            ValueError: If 'safe' is True, only permits lossless casting.

        """
        if cls.is_Z(value):
            return value
        elif (casted := int(value)) == value:
            return casted

        if safe:
            raise ValueError(f"'{value}' does not belong to integer set")

        warnmsg = f"Casted '{value}' to '{casted}' (integer set)"
        warn(warnmsg, RuntimeWarning)
        return casted

    @classmethod
    def cast_N(cls, value, /, *, safe: bool = False) -> int:
        """
        Return natural number.

        Raises:
            ValueError: If 'safe' is True, only permits lossless casting.

        """
        if (casted := cls.cast_Z(value)) >= 0:
            return casted

        if safe:
            raise ValueError(f"'{value}' does not belong to natural set")

        warnmsg = f"Casted '{value}' to '{casted}' (natural set)"
        warn(warnmsg, RuntimeWarning)
        return casted


__all__ += [Hint.__name__, cast2numpy.__name__, isNone.__name__,
            collapse_array.__name__, NumberSets.__name__]


# ============================================================================ #
# Pathing and files
# ---------------------------------------------------------------------------- #
def get_parent(path):
    """Given a filepath, return a path one directory higher."""
    parent_path, _ = os.path.split(path)
    return parent_path


def get_child(path):
    """Given a filepath, return the lowest level directory in the path name."""
    _, child_path = os.path.split(path)
    return child_path


class PathAnchor:
    """
    Instantiate to spawn an "anchor" in your code, from which various useful
    absolute paths relating to the anchor's location can be derived.
    """

    def __init__(self):
        self._stack = inspect.stack()

    @property
    def filepath(self) -> str:
        """Returns the path of file in which the anchor was spawned."""
        return self._stack[1].filename

    @property
    def filename(self) -> str:
        """
        Returns the name of the file in which the anchor was spawned, including
        any file extensions.
        """
        return get_child(path=self.filepath)

    @property
    def filename_stem(self) -> str:
        """Returns the stem of the filename (no extensions)."""
        stem, = re.findall("(.+)(?=[^.]*\.)+", self.filename)
        return stem

    @property
    def filename_ext(self) -> str:
        """Returns the extension of the filename, with the dot."""
        return self.filename[len(self.filename_stem):]

    @property
    def directory_path(self) -> str:
        """
        Returns the path to the directory containing the file in which the
        anchor was spawned.
        """
        return get_parent(path=self.filepath)

    @property
    def library_path(self) -> str:
        """
        Returns, if appropriate, the path to the library of code in which the
        anchor was spawned.

        Notes:
            This only works if folders are explicitly identified as modules
            using an "__init__.py" file - which is not required in versions of
            Python >= 3.4.

        """
        ascended = False
        current_path = self.directory_path

        # Look for explicit module declarations. When they stop, we stop.
        while "__init__.py" in os.listdir(current_path):
            ascended = True
            current_path = get_parent(current_path)

        if ascended is False:
            error_msg = (
                f"{type(self).__name__} does not appear to have spawned inside "
                f"an explicitly defined module - a more appropriate call would "
                f"be made with 'self.{type(self).library_path.fget.__name__}'"
            )
            warn(message=error_msg, category=RuntimeWarning)

        return current_path

    @classmethod
    @property
    def home_path(cls):
        path = os.path.expanduser("~")
        return path


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


__all__ += [PathAnchor.__name__, LoadData.__name__]


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


def revert2scalar(func):
    @functools.wraps(func)
    def with_reverting(*args, **kwargs):
        """Try to turn x into a scalar (if it is an array, list, or tuple)."""

        # Evaluate the wrapped func
        output = func(*args, **kwargs)

        if not isinstance(output, tuple):
            output = (output,)

        # Convert all items in the output to scalar if possible
        new_output = []
        for x in output:
            if isinstance(x, np.ndarray):
                if x.ndim == 0:
                    new_output.append(x.item())
                    continue
                if sum(x.shape) == 1:
                    new_output.append(x[0])
                    continue
            elif isinstance(x, (list, tuple)):
                if len(x) == 1:
                    new_output.append(x[0])
                    continue
            new_output.append(x)

        # If there was only one output from the function, return that as scalar
        if len(new_output) == 1:
            return new_output[0]

        return tuple(new_output)

    return with_reverting


__all__ += [call_count.__name__, call_depth.__name__, revert2scalar.__name__]
