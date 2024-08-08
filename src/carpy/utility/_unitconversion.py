"""A module supporting the intelligent conversion between systems of units."""
from functools import partial
import os
import re

import numpy as np
import polars as pl

from carpy.utility._maths import RationalNumber
from carpy.utility._miscellaneous import PathAnchor

__all__ = ["UnitOfMeasurement", "Quantity"]
__author__ = "Yaseen Reza"

# Load data for quantity representation
anchor = PathAnchor()
filename = "quantities.xlsx"
filepath = os.path.join(anchor.directory_path, "data", filename)
dataframes = pl.read_excel(filepath, sheet_id=0)
dataframes["dimensions"] = dataframes["dimensions"].with_row_index("id")  # Add a reliable way to get original id
# The "null" prefix should be treated as an empty string and not missing data lol
dataframes["prefixes"] = dataframes["prefixes"].with_columns(pl.col("Symbol").fill_null(pl.lit("")))


class UnitOfMeasurement:
    """
    A unit object is used to record the dimensionality, scale, and other properties of a unit.

    A simple set of maths operators are overloaded in this class to allow users to compare the dimensionality of one
    unit to another. For example, an object of this class representing horsepower will cleanly divide by another
    representing watts and return a dimensionless unit of measurement object. This does mean that class data on the
    original arguments of instantiation are lost, however, this is by design and believed to be more useful.
    """
    _memo_ids = {}
    _memo_dims = {}
    _memo_func = {}
    _re = re.compile(r"([^^]+)(?:\^(?<=\^){?([^{}]+)}?)?")
    _sibase = ["kg", "m", "s", "A", "K", "mol", "cd"]  # MKS (metre, kilogram, second) system of quantities
    _si_ext = _sibase + ["rad", "sr"]  # angular quantities included
    _symbols_powers: dict

    def __new__(cls, symbols: str = None, /):
        """
        Return an instance of the class, and memoize the unit pattern.

        Args:
            symbols: Symbols that represent units and measurement quantities.

        """
        if symbols is None or symbols == "":
            return super(UnitOfMeasurement, cls).__new__(cls)  # Nothing to do but return right away

        # Identify each constituent symbol with logical comparisons to defined units and systems in the dataframes
        for symbol in symbols.split():

            # If we already know what the symbol is supposed to be, don't bother with the polars dataframe lookup
            if symbol in cls._memo_ids:
                continue

            # Valid prefixes don't constitute the entirety of the symbol being looked at
            valid_prefixes = {
                id: prefix for (id, prefix) in enumerate(dataframes["prefixes"]["Symbol"])
                if (
                        symbol.startswith(prefix) and  # The symbol must start with the prefix
                        symbol != prefix and  # The symbol must not be the prefix
                        re.match(rf"{prefix}(?!\^)", symbol)  # The symbol must not be directly followed by the power
                )
            }

            # Given the prefix, what is the inferred suffix and inferred system?
            inferred_combinations = [
                (prefix_id, re.match(rf"{prefix}(\w+)", symbol).groups()[0], prefix_system)
                for (prefix_id, prefix) in valid_prefixes.items()
                for prefix_system in (
                    dataframes["prefixes"]["System"][i]
                    for (i, x) in enumerate(dataframes["prefixes"]["Symbol"] == prefix)
                    if x  # x is a boolean
                )
            ]
            # The below code is commented out because while filter is a recommended method, its 10x slower than tuple
            # comprehension for some reason right now...
            # inferred_combinations = [
            #     (prefix_id, re.match(rf"{prefix}(\w+)", symbol).groups()[0], prefix_system)
            #     for (prefix_id, prefix) in valid_prefixes.items()
            #     for prefix_system in dataframes["prefixes"].filter(pl.col("Symbol") == prefix)["System"]
            # ]

            # Valid suffixes are the actual units of quantity measurement
            valid_combinations = []
            for (prefix_id, inferred_suffix, prefix_system) in inferred_combinations:

                valid_suffixes = {
                    id: inferred_suffix for (id, symbols) in enumerate(dataframes["dimensions"]["Symbol(s)"])
                    # If the inferred suffix is actually in the Symbol(s) and
                    if inferred_suffix in symbols.split(",") and (
                        # Filter by the known system. If I don't know, it's definitely not Bel (that *needs* a prefix!)
                        (dataframes["dimensions"]["System"][id] == prefix_system) if prefix_system is not None
                        else (dataframes["dimensions"]["System"][id] != "Bel")
                    )
                }

                # Below code is also commented out because it appears to be slow when using polars filters...
                # # If the prefix inferred the use of a system, restrict the search space of valid units to that system
                # dimension_table = dataframes["dimensions"]
                # if prefix_system is not None:
                #     dimension_table = dimension_table.filter(pl.col("System") == prefix_system)
                # elif prefix_system != "Bel":
                #     dimension_table = dimension_table.filter(pl.col("System") != "Bel")
                #
                # # Tabulate suffixes
                # valid_suffixes = {
                #     dimension_table["id"][reduced_index]: inferred_suffix
                #     for (reduced_index, suffix) in enumerate(dimension_table["Symbol(s)"])
                #     if inferred_suffix in suffix.split(",")
                # }

                for suffix_id in valid_suffixes:
                    valid_combinations.append((prefix_id, suffix_id))

            if len(valid_combinations) == 0:
                error_msg = f"Could not determine what unit of measurement {symbol} means in {' '.join(symbols)}"
                raise ValueError(error_msg)

            elif len(valid_combinations) > 1:
                error_msg = f"Could not deconflict multiple potential meanings of the unit of measurement {symbol}"
                raise RuntimeError(error_msg)

            # Unpack and memoise the ids (prefix_id, suffix_id)
            (prefix_id, suffix_id), = valid_combinations
            (symbol_no_power, _), = cls._re.findall(symbol)
            cls._memo_ids[symbol_no_power], = valid_combinations

            # Memoize an array representing dimensionality
            dimension_row = dataframes["dimensions"][suffix_id]
            cls._memo_dims[symbol_no_power], = dimension_row[cls._si_ext].to_numpy()

            # Memoize conversions to and from SI
            prefix_exp = dataframes["prefixes"][prefix_id]["Exponent"].item()
            if dimension_row["System"].item() == "SI":
                if dimension_row["Symbol(s)"].item() == "g":
                    prefix_exp -= 3  # Naturally, correct for the fact that "g" is not SI but "kg" is!

                def to_si(value, symbol_power, prefix_exp):
                    return value * (10.0 ** prefix_exp) ** symbol_power

                def from_si(value, symbol_power, prefix_exp):
                    return value / (10.0 ** prefix_exp) ** symbol_power

                cls._memo_func[symbol_no_power] = {
                    "to_si": partial(to_si, prefix_exp=prefix_exp),
                    "from_si": partial(from_si, prefix_exp=prefix_exp)
                }

            elif dimension_row["System"].item() == "Bel":
                assert symbol == symbols, f"A bel system unit may not be instantiated in a compound unit like {symbols}"

                if {"f", "m", "W", "k"} & set(dimension_row["Symbol(s)"].item().split(",")):
                    offset = dimension_row["factor"].item()

                    def to_si(value, symbol_power, dBshift):
                        return 10.0 ** ((value + dBshift) / 10.0)

                    def from_si(value, symbol_power, dBshift):
                        return (10.0 * np.log10(value)) - dBshift

                    cls._memo_func[symbol_no_power] = {
                        "to_si": partial(to_si, dBshift=offset),
                        "from_si": partial(from_si, dBshift=offset)
                    }

                else:
                    error_msg = f"unit conversions for {symbol} in {symbols} are unsupported at this time"
                    raise NotImplementedError(error_msg)

            else:
                factor = dimension_row["factor"].item()

                def to_si(value, symbol_power, prefix_exp, factor):
                    return value * factor * (10.0 ** prefix_exp) ** symbol_power

                def from_si(value, symbol_power, prefix_exp, factor):
                    return value / factor / (10.0 ** prefix_exp) ** symbol_power

                cls._memo_func[symbol_no_power] = {
                    "to_si": partial(to_si, prefix_exp=prefix_exp, factor=factor),
                    "from_si": partial(from_si, prefix_exp=prefix_exp, factor=factor)
                }

        return super(UnitOfMeasurement, cls).__new__(cls)

    def __init__(self, symbols: str = None, /):
        """
        Args:
            symbols: Symbols that represent units and measurement quantities.
        """
        # Spawn a fresh dictionary for tracking symbols and their units
        self._symbols_powers = dict()

        if symbols is None or symbols == "":
            return  # Nothing to be done, just leave

        # Record the exponent associated with the units
        for symbol in symbols.split():
            (symbol_no_power, power), = self._re.findall(symbol)
            power = RationalNumber(*map(float, power.split("/"))) if power else 1
            if symbol_no_power in self._symbols_powers:
                self._symbols_powers[symbol_no_power] += power
            else:
                self._symbols_powers[symbol_no_power] = power

        # Sort the symbol and power dictionary
        def symbol_sorter(sym):
            sym_dims = self._memo_dims[sym]
            score_qtyidx = np.argmax(sym_dims != 0)  # Score by first appearance of which quantity
            score_exp = -self._symbols_powers[sym] * sym_dims[score_qtyidx]  # Exponent of first nonzero dim
            return score_exp, score_qtyidx

        self._symbols_powers = {
            k: self._symbols_powers[k] for k in sorted(self._symbols_powers, key=symbol_sorter)
            if self._symbols_powers[k] != 0  # Scrub any units that have a power of zero
        }
        return

    def __repr__(self):
        units, = self.args
        return units

    def __eq__(self, other):
        """Equality should hold true if dimensionality is equal."""
        cls = type(self)
        if not isinstance(other, cls):
            error_msg = f"Illegal operation, only {cls.__name__} objects may be compared (got {type(other).__name__})"
            raise TypeError(error_msg)
        raise NotImplementedError

    def __mul__(self, other):
        """Multiplication of powers means adding them together."""
        cls = type(self)
        if not isinstance(other, cls):
            error_msg = f"Illegal operation, only {cls.__name__} objects may be multiplied (got {type(other).__name__})"
            raise TypeError(error_msg)

        new_dims = self.dims + other.dims
        new_arg = " ".join([f"{self._si_ext[i]}^{dim_power}" for (i, dim_power) in enumerate(new_dims) if dim_power])
        new_obj = cls(new_arg)
        return new_obj

    def __pow__(self, power, modulo=None):
        """Units raised to a power means multiplying the units dimensional power by the encumbent power."""
        cls = type(self)
        if not isinstance(power, (int, float, np.floating, np.integer)):
            error_msg = f"{cls.__name__} must be raised to the power of a numeric type, not {type(power).__name__} type"
            raise TypeError(error_msg)

        # Unit raised to nan power should return a dead unit
        if np.isnan(power):
            return cls(None)

        new_dims = self.dims * power
        new_arg = " ".join([f"{self._si_ext[i]}^{dim_power}" for (i, dim_power) in enumerate(new_dims) if dim_power])
        new_obj = cls(new_arg)
        return new_obj

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        """Division of powers means subtracting other powers from self."""
        cls = type(self)
        if not isinstance(other, cls):
            error_msg = f"Illegal operation, only {cls.__name__} objects may be added (got {type(other).__name__})"
            raise TypeError(error_msg)

        new_dims = self.dims - other.dims
        new_arg = " ".join([f"{self._si_ext[i]}^{dim_power}" for (i, dim_power) in enumerate(new_dims) if dim_power])
        new_obj = cls(new_arg)
        return new_obj

    @property
    def dims(self) -> np.ndarray:
        """
        Return an array, where each term represents the dimensional power of a quantity.

        In order, each index of the output array corresponds with the base dimensional units listed in self._si_ext.
        """
        if not self._symbols_powers:
            return np.zeros(len(self._si_ext))

        stacked_dims = np.vstack([self._memo_dims[k] * v for (k, v) in self._symbols_powers.items()]).sum(axis=0)
        return stacked_dims

    @property
    def args(self) -> tuple[str]:
        """An args tuple that could be used to instantiate a new object with identical units."""
        instantiable_string = " ".join([
            f"{symbol}" if power == 1 else f"{symbol}^{power}"
            for (symbol, power) in self._symbols_powers.items()
        ])
        return (instantiable_string,)

    @property
    def is_dimensionless(self) -> bool:
        """Returns true if representing a dimensionless quantity. Radians are ratios by definition and ignored here."""
        if np.all(self.dims[:len(self._sibase)] == 0):
            return True
        return False

    def to_si(self, values):
        """Convert values in the instance's units of measure, to the equivalent values in SI equivalent base units."""
        values = np.atleast_1d(values)

        # Map imperial temperatures, if appropriate
        if self.args[0] == "degF":
            values = values - 32

        # In general
        for (symbol, power) in self._symbols_powers.items():
            transfer_func = self._memo_func[symbol].get("to_si")
            if not callable(transfer_func):
                raise NotImplementedError("Cannot convert unit to SI equivalent, no function exists")
            if symbol in self._si_ext:
                continue
            values = np.where(
                transfer_func(1, 1) != 1,
                transfer_func(values, power), values
            )

        # Map non-SI temperatures, if appropriate
        if self.args[0] in ["degC", "degF"]:
            values = values + 273.15

        return values

    def to_uom(self, values):
        """Convert values in SI units of measure, to the equivalent values in the instance's own units."""
        values = np.atleast_1d(values)

        # Map non-SI temperatures, if appropriate
        if self.args[0] in ["degC", "degF"]:
            values = values - 273.15

        # In general
        for (symbol, power) in self._symbols_powers.items():
            transfer_func = self._memo_func[symbol].get("from_si")
            if not callable(transfer_func):
                raise NotImplementedError("Cannot convert unit from SI equivalent, no function exists")
            if symbol in self._si_ext:
                continue
            values = np.where(
                transfer_func(1, 1) != 1,
                transfer_func(values, power), values
            )

        # Map imperial temperatures, if appropriate
        if self.args[0] == "degF":
            values = values + 32

        return values

    @property
    def units_si(self) -> str:
        dim_powers = self.dims
        si_string = " ".join([
            f"{symbol}" if power == 1 else f"{symbol}^{power}"
            for (symbol, power) in zip(self._si_ext, dim_powers)
            if power != 0
        ])
        return si_string


class Quantity(np.ndarray):
    _carpy_units: UnitOfMeasurement

    # =========================
    # numpy array compatibility
    # -------------------------

    def __new__(cls, values, /, units=None):
        # Recast the values to a numpy array
        values = np.atleast_1d(values)

        # If values were unicode or string, attempt to recast as float
        if values.dtype.kind in {"U", "S"}:
            try:
                values = values.astype(np.float64)
            except ValueError:
                error_msg = f"Expected numerical values for {cls.__name__} object (got {values.dtype.type} type)"
                raise ValueError(error_msg)

        # From the units, determine the SI equivalent quantity representation
        if isinstance(units, UnitOfMeasurement):
            unit_of_measurement = units
        else:
            unit_of_measurement = UnitOfMeasurement(units)
        values_SI = unit_of_measurement.to_si(values)

        # Subclass ourselves to np.ndarray
        obj = values_SI.view(cls)
        obj._carpy_units = unit_of_measurement
        return obj

    def __array_finalize__(self, obj):
        """
        References:
            https://numpy.org/doc/stable/user/basics.subclassing.html

        """
        if obj is None:
            return
        self._carpy_units = getattr(obj, "_carpy_units", None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        References:
            https://numpy.org/doc/stable/user/basics.subclassing.html

        """
        # If types of self are detected in the inputs, swap it for the original np.ndarray type
        args = []
        for input_ in inputs:
            if isinstance(input_, Quantity):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        # Same for the outputs (?)
        outputs = out
        if outputs:
            out_args = []
            for output in outputs:
                if isinstance(output, Quantity):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        # Compute the results
        results = getattr(ufunc, method)(*args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(
            (np.asarray(result).view(Quantity) if output is None else output)
            for result, output in zip(results, outputs)
        )

        # Finally, reintroduce the units
        for i, result in enumerate(results):

            if not isinstance(results[i], Quantity):
                continue  # Only bother with Quantity objects

            if ufunc.__name__ in ["arcsin", "arccos", "arctan", "arctan2"]:
                results[i]._carpy_units = UnitOfMeasurement("rad")
            else:
                results[i]._carpy_units = self.u

        return results[0] if len(results) == 1 else results

    # ==================================
    # Mathematical built-ins overloading
    # ----------------------------------

    def __abs__(self):
        """Absolute value."""
        return super(Quantity, self).__abs__()

    def __add__(self, other):
        """Addition."""
        cls = type(self)
        if isinstance(other, cls):
            assert np.all(self.u.dims == other.u.dims), f"Cannot add arrays with units {self.u} and {other.u}"
        return super(Quantity, self).__add__(other)

    def __ceil__(self):
        """Ceiling function."""
        return np.ceil(self)

    def __divmod__(self, other):
        """Division quotient, remainder."""
        return self.__floordiv__(other), self.__mod__(other)

    def __eq__(self, other):
        """Equality."""
        cls = type(self)
        if isinstance(other, cls):
            if self.u.dims != other.u.dims:
                return False
        return self.x == other

    def __float__(self):
        """Cast as float."""
        return float(self.x)

    def __floor__(self):
        """Floor function."""
        return np.floor(self)

    def __floordiv__(self, other):
        """self // other, floored division quotient."""
        cls = type(self)
        if isinstance(other, cls):
            new_value = self.x // other.x
            new_units = self.u / other.u
            return cls(new_value, new_units)
        return super(Quantity, self).__floordiv__(other)

    def __ge__(self, other):
        """Greater than or equal to."""
        cls = type(self)
        if isinstance(other, cls):
            assert np.all(self.u.dims == other.u.dims), f"Cannot compare arrays with units {self.u} and {other.u}"
        return super(Quantity, self).__ge__(other)

    def __gt__(self, other):
        """Greater than."""
        cls = type(self)
        if isinstance(other, cls):
            assert np.all(self.u.dims == other.u.dims), f"Cannot compare arrays with units {self.u} and {other.u}"
        return super(Quantity, self).__gt__(other)

    def __int__(self):
        """Cast as integer."""
        return int(self.x)

    def __le__(self, other):
        """Less than or equal to."""
        cls = type(self)
        if isinstance(other, cls):
            assert np.all(self.u.dims == other.u.dims), f"Cannot compare arrays with units {self.u} and {other.u}"
        return super(Quantity, self).__le__(other)

    def __lt__(self, other):
        """Less than."""
        cls = type(self)
        if isinstance(other, cls):
            assert np.all(self.u.dims == other.u.dims), f"Cannot compare arrays with units {self.u} and {other.u}"
        return super(Quantity, self).__lt__(other)

    def __mod__(self, other):
        """Modulo."""
        cls = type(self)
        if isinstance(other, cls):
            new_value = self.x - (self.x // other.x) * other.x
            new_units = self.u / other.u
            return cls(new_value, new_units)
        return super(Quantity, self).__mod__(other)

    def __mul__(self, other):
        """Multiplication."""
        cls = type(self)
        if isinstance(other, cls):
            new_value = self.x * other.x
            new_units = self.u * other.u
            return cls(new_value, new_units)
        return super(Quantity, self).__mul__(other)

    def __neg__(self):
        """Negation."""
        return super(Quantity, self).__neg__()

    def __pow__(self, power):
        """Self raised to a power."""
        cls = type(self)
        if isinstance(power, cls):
            assert power.u.is_dimensionless, f"If acting as an exponent, {type(self).__name__} must be dimensionless"
            power = power.x  # Squash the Quantity object, deal with array hereforward

        # By this point we establish that Quantity has one unit of measurement, and power has none.
        # If there is only one value in power, return type is a single Quantity object
        xs, powers = np.broadcast_arrays(self.x, power)
        nan_idxs = np.isnan(powers)
        if len(set(powers[~nan_idxs])) == 1:
            new_value = self.x ** power
            new_units = self.u ** powers[~nan_idxs].flat[0]
            return cls(new_value, new_units)

        # Else, an array of Quantities with different powers
        out = np.empty(xs.shape, dtype=np.ndarray)
        for i in range(out.size):
            new_value = xs.flat[i] ** powers.flat[i]
            new_units = self.u ** powers.flat[i]
            out.flat[i] = cls(new_value, new_units)
        return out

    def __radd__(self, other):
        """Reverse addition."""
        return self.__add__(other)

    def __rdivmod__(self, other):
        """Reverse divmod."""
        raise NotImplementedError

    def __rfloordiv__(self, other):
        """Reverse floor division."""
        raise NotImplementedError

    def __rmod__(self, other):
        """Reverse modulo."""
        raise NotImplementedError

    def __rmul__(self, other):
        """Reverse multiplication."""
        return self.__mul__(other)

    def __round__(self, n=None):
        """Round function."""
        return np.round(self, decimals=n)

    def __rpow__(self, other):
        """Reverse raise power."""
        raise NotImplementedError

    def __rsub__(self, other):
        """Reverse subtraction."""
        return -self.__sub__(other)

    def __rtruediv__(self, other):
        """Reverse true division."""
        return (self.__truediv__(other)) ** -1

    def __sub__(self, other):
        """Subtraction."""
        cls = type(self)
        if isinstance(other, cls):
            assert np.all(self.u.dims == other.u.dims), f"Cannot subtract arrays with units {self.u} and {other.u}"
        return super(Quantity, self).__sub__(other)

    def __truediv__(self, other):
        """True division."""
        cls = type(self)
        if isinstance(other, cls):
            new_value = self.x / other.x
            new_units = self.u / other.u
            return cls(new_value, new_units)
        return super(Quantity, self).__truediv__(other)

    def __trunc__(self):
        """Truncate."""
        return np.trunc(self)

    # =================================================
    # Other methods not directly interacting with numpy
    # -------------------------------------------------

    def __repr__(self):
        """For maths purposes, display SI."""
        repr1 = super(Quantity, self).__repr__()
        repr2 = self._carpy_units.units_si
        repr2 = "no_unit" if repr2 == "" else repr2
        return f"{repr1.rstrip()[:-1]}, {repr2})"

    def __str__(self):
        """For display purposes, display whatever the original unit was."""
        rtn_str1 = self._carpy_units.to_uom(self.x).__str__()
        rtn_str2, = self._carpy_units.args
        if rtn_str2:
            return f"{rtn_str1} {rtn_str2}"
        return rtn_str1

    def to(self, units: str) -> np.ndarray:
        """Return a numpy array in the chosen units."""
        new_uom = UnitOfMeasurement(units)
        new_value = new_uom.to_uom(self.x)
        return new_value

    @property
    def u(self) -> UnitOfMeasurement:
        """Return the units of measurement integral to the definition of the Quantity."""
        return self._carpy_units

    @property
    def x(self) -> np.ndarray:
        """Return the values of the internal numpy array without the Quantity class wrapper."""
        return np.array(self)
