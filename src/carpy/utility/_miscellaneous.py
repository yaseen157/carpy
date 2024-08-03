"""A module of miscellaneous library utilities."""
import functools
import inspect
import os
import re
from warnings import warn

import numpy as np
import pandas as pd
import yaml

__all__ = []
__author__ = "Yaseen Reza"


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


class classproperty(property):  # noqa: Ignore complaints about case, to make the style of decorator match @property
    """
    A decorator to use in place of '@property' when the property should be accessible from the class level.

    Examples:

        >>> class Foo:
        ...
        ...     @classproperty
        ...     def bar(self):
        ...         return "baz"

        >>> assert Foo().bar == "baz", "Couldn't locate instance property 'baz'"
        >>> assert Foo.bar == "baz", "Couldn't locate class property 'baz'"



    References:
        https://stackoverflow.com/questions/128573/using-property-on-classmethods

    """

    def __get__(self, owner_self, owner_cls):
        """Returns an attribute of instance. I don't know why this works. ~ Yaseen"""
        return self.fget(owner_cls)


__all__ += [call_count.__name__, call_depth.__name__, classproperty.__name__]


# ============================================================================ #
# Data types and casting
# ---------------------------------------------------------------------------- #

def is_none(*args) -> tuple:
    """True or False for every arg that is None."""
    results = tuple(map(lambda x: x is None, args))
    return results if len(args) > 1 else results[0]


class NumberSets:
    """
    A class of methods for testing whether a value belongs to a number set (e.g. real, complex), or for casting values
    as datatypes for said sets.
    """

    @staticmethod
    def is_complex(value, /) -> bool:
        """True if value is of a Python type that allows complex numbers."""
        return np.iscomplex(value)

    @staticmethod
    def is_real(value, /) -> bool:
        """True if value is of a Python type that allows real numbers."""
        return np.isfinite(value)

    @staticmethod
    def is_integer(value, /) -> bool:
        """True if value is of a Python type that allows integers."""
        return isinstance(value, (int, np.integer))

    @classmethod
    def cast_real(cls, value, /) -> float:
        """Return real number."""
        if cls.is_real(value):
            return value
        raise ValueError(f"Couldn't cast '{value}' to float (real number set)")

    @classmethod
    def cast_integer(cls, value, /, *, safe: bool = False) -> int:
        """
        Return integer number.

        Raises:
            ValueError: If 'safe' is True, only permits lossless casting.

        """
        if cls.is_integer(value):
            return value
        elif (casted := int(value)) == value:
            return casted

        if safe:
            raise ValueError(f"'{value}' does not belong to integer set")

        warnmsg = f"Casted '{value}' to '{casted}' (integer set)"
        warn(warnmsg, RuntimeWarning)
        return casted

    @classmethod
    def cast_natural(cls, value, /, *, safe: bool = False) -> int:
        """
        Return natural number.

        Raises:
            ValueError: If 'safe' is True, only permits lossless casting.

        """
        if (casted := cls.cast_integer(value)) >= 0:
            return casted

        if safe:
            raise ValueError(f"'{value}' does not belong to natural set")

        warnmsg = f"Casted '{value}' to '{casted}' (natural set)"
        warn(warnmsg, RuntimeWarning)
        return casted


def broadcast_vector(values, vector) -> tuple[np.ndarray, np.ndarray]:
    """
    Broadcast copies of array of values over each element of a vector.

    Args:
        values: An n-dimensional array of values with any shape.
        vector: A 1-dimensional array of values with shape (n,).

    Returns:
        A tuple of broadcasted values and vectors such that values has the shape (n, *values.shape), and the vector
            assumes the shape (n, *(1, ...)). Both the broadcasted arrays have the same number of dimensions making it
            easy to check each element in the vector against all values.

    Examples:

        >>> A = np.array([[1, 7], [3, -4]])
        >>> b = np.array([5, 0])

        # Broadcast copies of A over each element in vector b.
        >>> A_broadcasted, b_broadcasted = broadcast_vector(A, b)

        # Check which elements of A are greater than each element in b
        >>> print(A_broadcasted > b_broadcasted)
            array([[[False,  True],     # A > b[0]
                    [False, False]],

                   [[ True,  True],     # A > b[1]
                    [ True, False]]])

    """
    # Cast to arrays as necessary
    values = np.atleast_1d(values)
    vector = np.atleast_1d(vector)
    assert vector.ndim == 1, f"Expected vector to be a 1d array (got {vector.ndim=})"

    values_broadcast = np.broadcast_to(values, shape=(*vector.shape, *values.shape))
    vector_broadcast = np.expand_dims(vector, tuple(range(values_broadcast.ndim - 1))).T
    return values_broadcast, vector_broadcast


__all__ += [is_none.__name__, NumberSets.__name__, broadcast_vector.__name__]


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
        stem, = re.findall(r"(.+)(?=[^.]*\.)+", self.filename)
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
                f"an explicitly defined package. Couldn't locate '__init__.py'!"
            )
            warn(message=error_msg, category=RuntimeWarning)

        return current_path

    @classproperty
    def home_path(self):
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
