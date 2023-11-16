"""A module for the conversion of units."""
import copy
import functools
import itertools
import os
import re
from typing import Union
import warnings

import numpy as np
import pandas as pd

from carpy.utility._miscellaneous import cast2numpy, GetPath, Hint
from carpy.utility._vanity import Unicodify

__all__ = ["Quantity", "cast2quantity"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Load data required by the module
# ---------------------------------------------------------------------------- #
def load_quantities_dfs():
    """Load and return the quantity dimensions and prefixes spreadsheets."""
    filepath = os.path.join(GetPath.localpackage(), "data", "quantities.xlsx")
    dataframes = pd.read_excel(io=filepath, sheet_name=None, na_filter=False)
    return dataframes


dfs = load_quantities_dfs()

# ============================================================================ #
# Unit Comprehension
# ---------------------------------------------------------------------------- #

# Patterns need to be sorted from longest to shortest element for regex
# This pattern collects all the possible prefixes carpy will recognise and parse
pattern_prefixes = "|".join([
    f"(?:{sym})" for sym in sorted(dfs["prefixes"]["Symbol"], key=len)[::-1]])


class Dimensions(object):
    """
    This class can be used to create and track the dimensionality of any
    supported units, and find the dimensionally equivalent SI-base counterpart.

    """
    # Order base units as per the MKS (metre, kilogram, second) system of units
    _sibase = np.array(["kg", "m", "s", "A", "K", "mol", "cd"])

    # This cache is *very* important to performance. pandas.series lookups in
    # this class are very slow. By saving __init__ outcomes to a cache, some
    # downstream functions see a 5-7x speed improvement! This cache does depend
    # on users sticking to a regular way of writing units though, for example,
    # 'm s^{-1}' and 's^{-1} m' are not equivalent. The hope is that users do
    # stick a regular way of writing units (due to inherent user preferences),
    # which may even coincide with units used by the library.
    _sessioncache = {}

    def __init__(self, units: str = None, /):
        """
        Args:
            units: A space delimited string designating the units to instantiate
                this object with. If the unit symbol is ambiguous, the unit
                should be followed by '.<unit-system>' i.e.: .SI, .metric, .USC,
                or .imperial tags.
        """
        self._unitstring = "" if units is None else units

        # Check cache for an easy answer
        # (use deepcopy because of dictionary memory persistence nonsense)
        if self._unitstring in self._sessioncache:
            self._pairings = copy.deepcopy(self._sessioncache[self._unitstring])
            return

        # Otherwise, break units into components and proceed
        units = self._unitstring.split()

        # Dictionary will collapse repeated units into singular elements
        pairings = dict(zip(units, [dict() for _ in units]))
        for i, key in enumerate(units):

            unit = units[i]

            # Assess the power of the unit's dimension, consuming the string
            if not pairings[key].get("power"):
                pairings[key]["power"] = 0.0
            # Have to iadd the power in case there was a similar term earlier
            powerterm = re.findall(r"\^{(-?[0-9.]+)}", unit)
            if powerterm:
                pairings[key]["power"] += float(powerterm[0])
                unit = unit[:unit.index("^")]
            else:
                pairings[key]["power"] += 1.0

            # If the unit's system is given, determine it here
            system = re.findall(r"\.(.+)", unit)
            if system:
                system, = system  # Unpack findall's list into a string
                unit = unit[:unit.index(".")]
                # Check that this unit doesn't already have a dissimilar system
                if pairings[key].get("system", system) != system:
                    errormsg = (
                        f"'{unit}' in '{self._unitstring}' has conflicting unit"
                        f" systems. Deconflict by explicitly stating the system"
                        f" of this unit"
                    )
                    raise ValueError(errormsg)
                pairings[key]["system"] = system

            # Look for prefixes in the unit
            df_prefix = dfs["prefixes"]
            pattern = f"^(?:{pattern_prefixes})"
            prefixes = dict(
                df_prefix[
                    np.array([
                        x in set([""] + re.findall(pattern, unit))
                        for (i, x) in enumerate(df_prefix["Symbol"])
                    ])
                ]["Symbol"]
            )
            # If the prefix belongs to the Bel system, only Bel symbols allowed
            if "dB" in prefixes.values():
                pairings[key]["system"] = "Bel"

            # Create a dictionary of symbols indexed by the dataframe row number
            df_dims = dfs["dimensions"]
            system = pairings[key].get("system")
            symbols = df_dims["Symbol(s)"]
            if system is not None:
                symbols = dict(symbols[df_dims["System"] == system])
            else:
                symbols = dict(symbols[df_dims["System"] != "Bel"])

            # Try to pair a prefix with a unit symbol that completes the unit
            products = [
                (prefix_id, symbol_id)
                for prefix_id, symbol_id in itertools.product(prefixes, symbols)
                if unit in [
                    f"{prefixes[prefix_id]}{x}"
                    for x in symbols[symbol_id].split(",")
                ] and not (  # Fail condition: SI prefix on non-SI unit
                        df_prefix["System"][prefix_id] == "SI"
                        and prefixes[prefix_id] != ""
                        and df_dims["System"][symbol_id] != "SI"
                )
            ]
            if len(products) > 1:
                matched = [
                    f"<{symbols[i]}>.{df_dims['System'][i]}"
                    for _, i in products
                ]
                errormsg = (
                    f"'{key}' is too ambiguous a declaration of a unit. "
                    f"Discovered potential matches to symbols: {matched}"
                )
                raise ValueError(errormsg)
            elif len(products) == 0:
                errormsg = (
                    f"Could not identify what unit '{key}' might be referring "
                    f"to in the compound unit '{self._unitstring}'."
                )
                raise ValueError(errormsg)

            # Otherwise all is good, save the result...
            product, = products
            _, sym_id = product
            symbol = df_dims["Symbol(s)"][sym_id].split(",")[0]

            pairings[key]["(prefix, unit)"] = product
            pairings[key]["symbol"] = symbol
            pairings[key]["system"] = df_dims["System"][sym_id]

        # Store and write to cache
        self._pairings = pairings
        self._sessioncache[self._unitstring] = copy.deepcopy(self._pairings)
        return

    def __repr__(self):
        return f"{type(self).__name__}('{self._unitstring}')"

    def __str__(self):
        return Unicodify.mathscript_safe(self.si_equivalent)

    def __mul__(self, other):
        # Add dimensional powers
        dict_a = {**self.si_dimensioned, **self.si_dimensionless}
        dict_b = {**other.si_dimensioned, **other.si_dimensionless}
        dict_sum = {
            k: dict_a.get(k, 0) + dict_b.get(k, 0)
            for k in set(dict_a) | set(dict_b)
        }
        # Create a new instance and overwrite hidden attributes
        unitstring = " ".join(
            f"{k}.SI" + "^{%f}" % v for k, v in dict_sum.items()
        )
        output = self.__class__(unitstring)
        for k in output._pairings:
            output._pairings[k]["power"] = dict_sum[k[:k.index(".")]]
        return output

    def __truediv__(self, other):
        # Subtract dimensional powers
        dict_a = {**self.si_dimensioned, **self.si_dimensionless}
        dict_b = {**other.si_dimensioned, **other.si_dimensionless}
        dict_sum = {
            k: dict_a.get(k, 0) - dict_b.get(k, 0)
            for k in set(dict_a) | set(dict_b)
        }
        # Create a new instance and overwrite hidden attributes
        unitstring = " ".join(
            f"{k}.SI" + "^{%f}" % v for k, v in dict_sum.items())
        output = self.__class__(unitstring)
        for k in output._pairings:
            output._pairings[k]["power"] = dict_sum[k[:k.index(".")]]
        return output

    def __pow__(self, power, modulo=None):

        myunits = {**self.si_dimensioned, **self.si_dimensionless}

        # Create a new instance and overwrite hidden attributes
        unitstring = " ".join(
            f"{k}.SI" + "^{%f}" % v for k, v in myunits.items())
        output = self.__class__(unitstring)
        for k in output._pairings:
            output._pairings[k]["power"] = myunits[k[:k.index(".")]] * power

        return output

    def __invert__(self):
        # Simply return true division result
        return self.__class__("").__truediv__(self)

    @property
    def og_components(self) -> dict:
        """A dictionary describing this object's constituent units."""
        return self._pairings

    @property
    def og_string(self) -> str:
        """The original string that instantiated this unit object."""
        return self._unitstring

    @functools.cached_property
    def si_dimensioned(self) -> dict:
        """Return a dictionary dimensionality of the unit in SI base units."""
        dims = self._sibase
        mydims = dict(zip(dims, np.zeros(len(dims))))

        # Iterate over each component unit of the units object
        for _, (_, pairinfo) in enumerate(self._pairings.items()):
            _, unitid = pairinfo["(prefix, unit)"]
            unitpower = pairinfo["power"]
            unitseries = dfs["dimensions"].loc[unitid]

            # Iterate over each SI dimension and compound the answer
            for baseunit in dims:
                mydims[baseunit] += unitseries[baseunit] * unitpower

        return mydims

    @functools.cached_property
    def si_dimensionless(self) -> dict:
        """Return a dictionary of non-dimensional components of the unit."""
        dims = np.array(["rad", "sr"])
        mydims = dict(zip(dims, np.zeros(len(dims))))

        # Iterate over each component unit of the units object
        for _, (_, pairinfo) in enumerate(self._pairings.items()):
            _, unitid = pairinfo["(prefix, unit)"]
            unitpower = pairinfo["power"]
            unitseries = dfs["dimensions"].loc[unitid]

            # Iterate over each SI dimension and compound the answer
            for baseunit in dims:
                mydims[baseunit] += unitseries[baseunit] * unitpower

        return mydims

    @property
    def si_equivalent(self) -> str:
        """Return a string of dimensions in SI equivalent units."""
        stringelements = []
        si_dimensions = {**self.si_dimensionless, **self.si_dimensioned}
        for _, (symbol, power) in enumerate(si_dimensions.items()):
            if power == 1:
                stringelements.append(symbol)
            elif power == 0:
                pass
            elif int(power) == power:
                stringelements.append(symbol + "^{%d}" % power)
            else:
                stringelements.append(symbol + "^{%.3f}" % power)

        return " ".join(stringelements)


class Quantity(np.ndarray):
    """
    The Quantity class can be used to maintian consistency in the manipulation
    of quantities, measures, and their units.

    This class inherits many properties of classic numpy arrays.
    """

    def __new__(cls, value: Hint.nums, /, units: Union[str, Dimensions] = None):
        """
        Args:
            value: Any number(s) that you would expect a numpy array to support.
            units: Any argument the 'Units' class of this module supports.
        """
        # Recast as necessary
        value = cast2numpy(value, dtype=np.float64)
        units = units if isinstance(units, Dimensions) else Dimensions(units)

        # If the value is a quantity, raise error to user on recasted dimensions
        if isinstance(value, Quantity) and (
                value.units.si_equivalent != units.si_equivalent
        ):
            errormsg = f"Quantity {value} had dimensions recast to {units}"
            raise ValueError(errormsg)

        # Try and convert the value to SI values
        if units.og_string == "degC":
            value = value + 273.15
        elif units.og_string == "degF":
            value = (value - 32) * (5 / 9) + 273.15
        else:
            for _, (_, pairinfo) in enumerate(units.og_components.items()):
                prefix_id, unit_id = pairinfo["(prefix, unit)"]

                prefix_system = dfs["prefixes"]["System"][prefix_id]

                # Different conversions depending on the system of the prefix
                if prefix_system == "SI":
                    value = value * (
                            10.0 ** dfs["prefixes"]["Exponent"][prefix_id]
                            * dfs["dimensions"]["transform"][unit_id]
                    ) ** pairinfo["power"]

                elif prefix_system == "Bel":
                    value = value + dfs["dimensions"]["transform"][unit_id]
                    value = 10.0 ** (value / 10)  # map dB scale to linear scale

                else:
                    raise NotImplementedError(f"No method for {prefix_system}")

        # Now that values are in SI format, we want units of a new obj to be too
        # ... identify components of a unit string based on SI dimensions
        newunit_components = [
            f"{k}.SI" + ("^{%f}" % v)
            for k, v in units.si_dimensioned.items()
        ]
        # ... identify components of a unit string based on angular dimensions
        newunit_components += [
            f"{k}.SI" + ("^{%f}" % v)
            for k, v in units.si_dimensionless.items()
        ]
        newunit_str = " ".join(newunit_components)

        # The new instance of this class should be an array object
        customarray = np.asarray(value).view(cls)
        # Add a custom attribute to the array object
        customarray.units = Dimensions(newunit_str)
        return customarray

    def __array_finalize__(self, obj, **kwargs):
        # This magic method is required for numpy compatibility
        if obj is None:
            return
        self.units = getattr(obj, "units", None)

    def __repr__(self):
        return f"{super().__repr__()[:-1]}, units='{self.units}')"

    def __format__(self, format_spec):
        if self.size == 1 or format_spec == "":
            pass  # Pass with conditional, allows me to add elifs down the line
        else:
            errormsg = "Only scalar quantities can be formatted (not arrays!)"
            raise ValueError(errormsg)
        si_units_utf8 = Unicodify.mathscript_safe(self.units.si_equivalent)
        return f"{format(self.x, format_spec)} {si_units_utf8}"

    def __str__(self):
        si_units_utf8 = Unicodify.mathscript_safe(self.units.si_equivalent)
        if self.size > 1:
            return f"{super().__str__()} {si_units_utf8}"
        return f"{super().__str__()[1:-1]} {si_units_utf8}"

    # --------------------- #
    # Relational Operations #
    # --------------------- #

    def _can_do_relational(self, other: object):
        """Check that it makes sense to compare two objects as Quantity objs."""
        if isinstance(other, self.__class__):
            dims0 = {**self.units.si_dimensionless,
                     **self.units.si_dimensioned}
            dims1 = {**other.units.si_dimensionless,
                     **other.units.si_dimensioned}
            if not (dims0 == dims1):
                # Two quantity objects with mismatched dimensions
                errormsg = (
                    f"Cannot carry out operation between {type(self).__name__} "
                    f"objects with dimensions do not match (got "
                    f"{self.units} and {other.units})"
                )
                raise ValueError(errormsg)
        return True

    def __lt__(self, other):
        if self._can_do_relational(other):
            return np.array(self) < other

    def __le__(self, other):
        if self._can_do_relational(other):
            return np.array(self) <= other

    def __eq__(self, other):
        if self._can_do_relational(other):
            return np.array(self) == other

    def __ne__(self, other):
        if self._can_do_relational(other):
            return np.array(self) != other

    def __gt__(self, other):
        if self._can_do_relational(other):
            return np.array(self) > other

    def __ge__(self, other):
        if self._can_do_relational(other):
            return np.array(self) >= other

    # --------------------- #
    # Arithmetic Operations #
    # --------------------- #

    def __add__(self, other):
        if self._can_do_relational(other):  # <-- Ensures matching units
            return super().__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if self._can_do_relational(other):  # <-- Ensures matching units
            return super().__sub__(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # For mismatched objects, use default behaviour
        if not isinstance(other, type(self)):
            return super().__mul__(other)

        # Otherwise 2x Quantity objects. Create and return a new object
        value = np.array(self) * np.array(other)
        units = self.units * other.units
        return self.__class__(value, units)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        warnmsg = (
            f"{type(self).__name__} objects do not yet support the '@' "
            f"operator. Collapsing result into regular numpy arrays w/o units"
        )
        warnings.warn(message=warnmsg, category=RuntimeWarning)
        return np.array(self).__matmul__(other)

    def __rmatmul__(self, other):
        warnmsg = (
            f"{type(self).__name__} objects do not yet support the '@' "
            f"operator. Collapsing result into regular numpy arrays w/o units"
        )
        warnings.warn(message=warnmsg, category=RuntimeWarning)
        return np.array(self).__rmatmul__(other)

    def __imatmul__(self, other):
        warnmsg = (
            f"{type(self).__name__} objects do not yet support the '@' "
            f"operator. Collapsing result into regular numpy arrays w/o units"
        )
        warnings.warn(message=warnmsg, category=RuntimeWarning)
        # noinspection PyUnresolvedReferences
        return np.array(self).__imatmul__(other)

    def __truediv__(self, other):
        # For mismatched objects, use default behaviour
        if not isinstance(other, type(self)):
            return super().__truediv__(other)

        # Otherwise 2x Quantity objects. Create and return a new object
        value = np.array(self) / np.array(other)
        units = self.units / other.units
        return self.__class__(value, units)

    def __rtruediv__(self, other):
        # For mismatched objects, cast both to Quantity objects.
        if not isinstance(other, type(self)):
            other = Quantity(other, "")

        # 2x Quantity objects. Create and return a new object
        return other.__truediv__(self)

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __floordiv__(self, other):
        # For mismatched objects, use default behaviour
        if not isinstance(other, type(self)):
            return super().__floordiv__(other)

        # Otherwise 2x Quantity objects. Create and return a new object
        value = np.array(self) // np.array(other)
        units = self.units / other.units
        return self.__class__(value, units)

    def __rfloordiv__(self, other):
        # For mismatched objects, cast both to Quantity objects.
        if not isinstance(other, type(self)):
            other = Quantity(other, "")

        # 2x Quantity objects. Create and return a new object
        return other.__floordiv__(self)

    def __ifloordiv__(self, other):
        return self.__floordiv__(other)

    def __mod__(self, other):
        # For mismatched objects, use default behaviour
        if not isinstance(other, type(self)):
            return super().__mod__(other)

        # Otherwise 2x Quantity objects. Create and return a new object
        value = np.array(self) - (np.array(self) // np.array(other))
        units = self.units / other.units
        return self.__class__(value, units)

    def __rmod__(self, other):
        # For mismatched objects, cast both to Quantity objects.
        if not isinstance(other, type(self)):
            other = Quantity(other, "")

        # 2x Quantity objects. Create and return a new object
        return other.__mod__(self)

    def __imod__(self, other):
        return self.__mod__(other)

    def __divmod__(self, other):
        return self.__floordiv__(other), self.__mod__(other)

    def __rdivmod__(self, other):
        # For mismatched objects, cast both to Quantity objects.
        if not isinstance(other, type(self)):
            other = Quantity(other, "")

        # 2x Quantity objects. Create and return a new object
        return other.__divmod__(self)

    def __pow__(self, power, modulo=None):

        if modulo is not None:
            return NotImplemented

        if isinstance(power, self.__class__) and str(power.units) != "":
            errormsg = (
                "Exponent cannot be a Quantity object with non-negligible units"
            )
            raise ValueError(errormsg)

        value = np.array(super().__pow__(power, modulo))
        units = self.units ** power
        return Quantity(value, units)

    def __rpow__(self, power):

        if str(self.units) != "":
            errormsg = (
                "Exponent cannot be a Quantity object with non-negligible units"
            )
            raise ValueError(errormsg)
        return super().__rpow__(power)

    def __ipow__(self, power):
        return self.__pow__(power)

    # ---------------------- #
    # Arithmetic Overloading #
    # ---------------------- #

    def __round__(self, n=None):
        return super().round(decimals=n if n is not None else 0)

    def __trunc__(self):
        return np.trunc(self)

    def __floor__(self):
        return np.floor(self)

    def __ceil__(self):
        return np.ceil(self)

    def __float__(self):
        flattened = self.flatten()
        # If Quantity only contains a single scalar value, take it!
        if len(flattened) == 1:
            return float(self.flat[0])  # *actually* cast it as a float
        # Otherwise we're losing information unintentionally, raise an error
        else:
            errormsg = "Conversion of multi-element array to scalar is lossy"
            raise ValueError(errormsg)

    # ----------------- #
    # Slice Overloading #
    # ----------------- #
    def __getitem__(self, key):
        return Quantity(super().__getitem__(key), self.units)

    @property
    def x(self) -> Union[float, np.ndarray]:
        """The raw value(s) of the array, without associated units."""
        try:
            return float(self)
        except ValueError:  # Error from converting size > 1 array to scalar
            return np.array(self)

    def to(self, units: str, /) -> np.ndarray:
        """
        Convert this quantity into a desired unit, provided that the units have
        compatible dimensions.

        Args:
            units: A space delimited string designating the units to instantiate
                this object with. If the unit symbol is ambiguous, the unit
                should be followed by '.<unit-system>' i.e.: .SI, .metric, .USC,
                or .imperial tags.

        Returns:
            The value of this quantity in the specified units.

        """
        # Recast as necessary
        units = units if isinstance(units, Dimensions) else Dimensions(units)

        errormsg = f"Incompatible units ({self.units} != {units})"
        assert self.units.si_equivalent == units.si_equivalent, errormsg

        # Special cases where the unit needs to shift the scale
        if len(units.og_components) == 1:
            if units.og_string == "degC":
                return self.x - 273.15
            elif units.og_string == "degF":
                return (self.x - 273.15) * (9 / 5) + 32

            # Special cases where the scale is logarithmic
            elif units.og_string.startswith("dB"):
                pairinfo = units.og_components[units.og_string]
                _, unit_id = pairinfo["(prefix, unit)"]
                transform = dfs["dimensions"]["transform"][unit_id]
                return 10 * np.log10(self.x) - transform
        else:
            for _, (_, pairinfo) in enumerate(units.og_components.items()):
                if pairinfo["system"] == "Bel":
                    raise ValueError("(deci)Bel conversion not recognised")

        # Converting to units which differ by a scalar multiple is as easy as...
        output = self / self.__class__(1, units)

        # If it was successful, the si_equivalent units should've cancelled
        if output.units.si_equivalent == "":
            return output.x

        # The units weren't successfully mapped due to inconsistent units
        errormsg = (
            f"Target '{units=}' do not share dimensionality with {self.units=}")
        raise ValueError(errormsg)


def cast2quantity(pandas_df: pd.DataFrame) -> dict:
    """
    Convert the contents of a dataframe into a dictionary of quantities.

    Args:
        pandas_df: A pandas dataframe object.

    Returns:
        A dictionary of quantities.

    """
    # Fill blank values with NaN
    pandas_df = pandas_df.mask(pandas_df == "")

    # Iterate over the columns in the dataframe
    output = dict()
    for column_name in pandas_df.columns:
        name_components = re.split(r"\s\[(.+)]", column_name)
        new_values = pandas_df[column_name].to_numpy()

        # If units are not present, just copy the values over to the output
        if len(name_components) == 1:
            output[column_name] = new_values
            continue

        # Otherwise, check that the column doesn't already exist in output
        output_col, units, _ = name_components
        if output_col not in output:
            try:
                output[output_col] = Quantity(new_values, units)
            except ValueError:  # Couldn't parse the unit that was detected
                output[column_name] = new_values  # Just copy the values over

        # If column name already exists, assume both new column and old column
        # are referring to the same quantity - just different units. Try to
        # update missing elements of the original column with the new column
        else:
            wherenan = np.isnan(output[output_col])
            output[output_col][wherenan] = Quantity(new_values, units)[wherenan]

    return output
