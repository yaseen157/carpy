"""Methods for prescribing geometry the indexing of 'stations'."""
import numpy as np

from carpy.utility import Hint, cast2numpy, collapse_array

__all__ = ["DiscreteIndex", "ContinuousIndex"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Support functions
# ---------------------------------------------------------------------------- #

def validate_keys(*keys: Hint.any) -> tuple[bool, ...]:
    """
    Assert that arguments (for station numbers) are all numeric.

    Args:
        *keys: Dictionary keys to check.

    Returns:
        A tuple the same length as the *keys argument, true if the key is valid
         (numeric in nature), and false otherwise.

    Notes:
        Be careful - any() or all() functions should be used to determine the
            appropriate course of action after evaluation with this function.

    """
    valid_keys = tuple([
        True if isinstance(x, Hint.num.__args__) else False
        for x in keys
    ])
    return valid_keys


def bad_key(key) -> None:
    """Raise an error, and tell the user what was wrong."""
    errormsg = f"Station numbering {key=} is not numeric in nature!"
    raise ValueError(errormsg)


def sort_dict(dictionary: dict) -> dict:
    """Sort dictionaries by their keys."""
    sorted_dict = dict(sorted(dictionary.items(), key=lambda x: x[0]))
    return sorted_dict


def sort_kvs(nestedlist: list) -> list:
    """Sort results of a 'list(dict.items())' call by the dictionary keys."""
    sorted_list = sorted(nestedlist, key=lambda x: x[0])
    return sorted_list


def parse_slice(keys: np.ndarray, slice_object: slice) -> tuple:
    """Extract customised keys bounded by a slice object."""
    # Define the starting key, if it is missing
    if (start := slice_object.start) is None:
        start = keys[0]

    # If stop key is missing, use <= (get everything). Else, < (up to stop key)
    if (stop := slice_object.stop) is None:
        stop = keys[-1]
        realkeys = keys[(start <= keys) & (keys <= stop)]
    else:
        realkeys = keys[(start <= keys) & (keys < stop)]

    # If no step is given, just return the relevant keys
    if (step := slice_object.step) is None:
        return realkeys

    # Else a "step" is given, create a linear spacing of keys with 'step' elems.
    # noinspection PyArgumentList
    return np.linspace(keys.min(), keys.max(), step)


# ============================================================================ #
# Classes for consistent indexing behaviour
# ---------------------------------------------------------------------------- #
class TransparentArray(list):
    """
    Class for broadcasting get and set attributes to items *within* an array
    (instead of trying to access attributes of the array itself).
    """

    def __getattr__(self, item):
        return [getattr(x, item) for x in self]

    def __setattr__(self, name, value):
        if (n := len(self)) == 0:
            raise ValueError(f"Couldn't broadcast {value} to empty slice")

        # Broadcast assignment
        values = np.broadcast_to(value, n)
        [setattr(x, name, y) for (x, y) in np.broadcast(self, values)]
        return


class DiscreteIndex(dict):
    """
    Template class for Station-based geometry numbering systems, with linear
    interpolation between a discrete number of prescribed geometries.
    """

    def __init__(self, mapping: dict = None, /, **kwargs):
        """
        Args:
            mapping: Initial mapping of keys and values.
            **kwargs: Updates to apply to the mapping.
        """
        # Recast as necessary
        mapping = dict() if mapping is None else mapping
        if kwargs:
            mapping.update(dict([(k, v) for (k, v) in kwargs.items()]))

        # Validate mapping (keys)
        remapped = dict()
        for (key, value) in mapping.items():

            if validate_keys(key)[0]:
                # Ensure key maps as float for consistency of operation
                remapped[float(key)] = value
                continue

            bad_key(key)  # <-- Raise error on bad key

        # Superclass call (with sorted version of self)
        super().__init__(sort_dict(remapped))
        return

    def __setitem__(self, key, value):

        if validate_keys(key):
            # Record current entries of the dictionary, and clear self
            items = list(self.items())
            self.clear()

            # Ensure key maps as float for consistency of operation
            items.append((float(key), value))

            # Sort entries, and place them back into self
            items = sort_kvs(items)
            [super(DiscreteIndex, self).__setitem__(*kv) for kv in items]
            return

        elif isinstance(key, slice):
            errormsg = (
                "Do not set values using a slice object. Instead, use "
                "individual keys as appropriate."
            )
            raise NotImplementedError(errormsg)
        #     # Record current entries of the dictionary, and clear self
        #     items = list(self.items())
        #     self.clear()
        #
        #     # Ensure key maps as float for consistency of operation
        #     items.append((float(key), value))

        bad_key(key)  # <-- Raise error on bad key

    def __getitem__(self, key):

        # Need to get the key if it exists, else interpolate the data...
        keys, vals = zip(*self.items())
        keys, vals = map(cast2numpy, [keys, vals])  # Make indexable

        # If slice object, find the keys being referenced - else, only one key
        if isinstance(key, slice):
            keys2get = parse_slice(keys, slice_object=key)
            sliced = True
        else:
            keys2get = cast2numpy(key)  # Make iterable
            sliced = False

        vals2rtn = list()
        for key in keys2get:
            # If key exists or is out of bounds, take the closest matching key
            if key in keys:
                vals2rtn.append(super().__getitem__(key))
            elif key > max(keys):
                vals2rtn.append(super().__getitem__(keys[-1]))
            elif key < min(keys):
                vals2rtn.append(super().__getitem__(keys[0]))
            # Otherwise, we have to interpolate between keys in self
            else:
                # Find bounding keys
                keyidx_hi = np.argmax(key < keys)
                key_hi = keys[keyidx_hi]
                key_lo = keys[keyidx_hi - 1]

                # Linearly interpolate contribution of values at the keys
                weight = (key - key_lo) / (key_hi - key_lo)
                val_hi = super().__getitem__(key_hi)
                val_lo = super().__getitem__(key_lo)
                val_new = val_hi * weight + val_lo * (1 - weight)
                vals2rtn.append(val_new)

        # If sliced, return an array
        if sliced is True:
            return TransparentArray(vals2rtn)
        return collapse_array(TransparentArray(vals2rtn))


class ContinuousIndex(object):
    """
    Template class for Station-based geometry numbering systems. Station-wise
    geometry is prescribed procedurally, and so the set of accessible stations
    is continuous.
    """

    def __init__(self, procedure: Hint.func):
        self._procedure = procedure
        return

    def __setitem__(self, key, value):
        errormsg = (
            f"Procedurally generated stations cannot be overridden at discrete "
            f"points. Perhaps you meant to use this on a subclass of "
            f"{DiscreteIndex.__name__} (instead of "
            f"{ContinuousIndex.__name__})"
        )
        raise AttributeError(errormsg)

    def __getitem__(self, key):
        if isinstance(key, slice):
            errormsg = (
                f"Cannot obtain stations using index slicing (got {key}). This "
                f"is because the stations are defined continuously with a "
                f"procedure (and are therefore not iterables that can be "
                f"sliced). Use discrete station positions instead."
            )
            raise LookupError(errormsg)

        # Recast as necessary
        keys2get = cast2numpy(key)
        vals2rtn = [self._procedure(x) for x in keys2get]

        return vals2rtn[0] if len(vals2rtn) == 1 else tuple(vals2rtn)

# ============================================================================ #
# Classes for Stations
# ---------------------------------------------------------------------------- #

# Each station numbering type will have a different set of unique properties
# for example, fuselage geometry are always oriented with the x-axis,
# buttock with y, etc.
# Fuselage: StationsLinear  # FS
# ButtockLine: StationsLinear  # BL
# WaterLine: StationsLinear  # WL
# Aileron: StationsLinear  # AS
# Flap: StationsLinear  # KS
# Nacelle: StationsLinear  # NC
# Wing: StationsLinear  # WS
# HorizontalStabiliser: StationsLinear  # HSS
# VerticalStabiliser: StationsLinear  # VSS
# Powerplant: StationsLinear  # PPS
