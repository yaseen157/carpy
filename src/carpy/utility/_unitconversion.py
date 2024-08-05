"""A module supporting the intelligent conversion between systems of units."""
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

# The "null" prefix should be treated as an empty string and not missing data lol
dataframes["prefixes"] = dataframes["prefixes"].with_columns(pl.col("Symbol").fill_null(pl.lit("")))


class UnitOfMeasurement:
    """A unit object is used to record the dimensionality, scale, and other properties of a unit."""
    _memo_ids = {}
    _memo_dims = {}
    _memo_conv = {}
    _re = re.compile(r"([^^]+)(?:\^(?<=\^){?([^{}]+)}?)?")
    _sibase = ["kg", "m", "s", "A", "K", "mol", "cd"]  # MKS (metre, kilogram, second) system of quantities
    _si_ext = _sibase + ["rad", "sr"]  # angular quantities included
    _symbols_powers: dict

    def __new__(cls, symbols: str = None, /):
        """
        Return an instance of the class, and memoize the unit pattern

        Args:
            symbols: Symbols that represent units and measurement quantities.

        """
        # Set as empty string to skip the coming for-loop
        if symbols is None:
            symbols = ""

        # Identify each constituent symbol with logical comparisons to defined units and systems in the dataframes
        for symbol in symbols.split(" "):

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
                for prefix_system in dataframes["prefixes"].filter(pl.col("Symbol") == prefix)["System"]
            ]

            # Valid suffixes are the actual units of quantity measurement
            valid_combinations = []
            for (prefix_id, inferred_suffix, prefix_system) in inferred_combinations:

                # If the prefix inferred the use of a system, restrict the search space of valid units to that system
                dimension_table = dataframes["dimensions"].with_row_index("id")  # Add a reliable way to get original id
                if prefix_system is not None:
                    dimension_table = dimension_table.filter(pl.col("System") == prefix_system)
                elif prefix_system != "Bel":
                    dimension_table = dimension_table.filter(pl.col("System") != "Bel")

                # Tabulate suffixes
                valid_suffixes = {
                    dimension_table["id"][reduced_index]: inferred_suffix
                    for (reduced_index, suffix) in enumerate(dimension_table["Symbol(s)"])
                    if inferred_suffix in suffix.split(",")
                }
                for suffix_id in valid_suffixes:
                    valid_combinations.append((prefix_id, suffix_id))
            else:
                # clean-up namespace
                del inferred_suffix, inferred_combinations
                del prefix_system, dimension_table, valid_prefixes, valid_suffixes

            if len(valid_combinations) == 0:
                error_msg = f"Could not determine what unit of measurement {symbol} means in {' '.join(symbols)}"
                raise ValueError(error_msg)

            elif len(valid_combinations) > 1:
                error_msg = f"Could not deconflict multiple potential meanings of the unit of measurement {symbol}"
                raise RuntimeError(error_msg)

            # Unpack and memoise the ids (prefix_id, suffix_id) and an array representing dimensionality
            (prefix_id, suffix_id), = valid_combinations
            dimension_row = dataframes["dimensions"][suffix_id]
            (symbol_no_power, _), = cls._re.findall(symbol)
            prefix_exp = dataframes["prefixes"][prefix_id]["Exponent"].item()
            cls._memo_ids[symbol_no_power], = valid_combinations
            cls._memo_dims[symbol_no_power], = dimension_row[cls._si_ext].to_numpy()

            # Memoize conversions to and from SI
            if dimension_row["System"].item() == "SI":

                def to_si(value, symbol_power):
                    return value * (10.0 ** prefix_exp) ** symbol_power

                def from_si(value, symbol_power):
                    return value / (10.0 ** prefix_exp) ** symbol_power

            elif dimension_row["System"].item() == "Bel":
                assert symbol == symbols, f"A bel system unit may not be instantiated in a compound unit like {symbols}"

                if {"f", "m", "W", "k"} & set(dimension_row["Symbol(s)"].item().split(",")):
                    dBshift = dimension_row["factor"].item()

                    def to_si(value, symbol_power):
                        return 10.0 ** ((value + dBshift) / 10.0)

                    def from_si(value, symbol_power):
                        return (10.0 * np.log10(value)) - dBshift

                else:
                    error_msg = f"unit conversions for {symbol} in {symbols} are unsupported at this time"
                    raise NotImplementedError(error_msg)

            else:
                factor = dimension_row["factor"].item()

                def to_si(value, symbol_power):
                    return value * factor * (10.0 ** prefix_exp) ** symbol_power

                def from_si(value, symbol_power):
                    return value / factor / (10.0 ** prefix_exp) ** symbol_power

            cls._memo_conv[symbol_no_power] = {"to_si": to_si, "from_si": from_si}

        return super(UnitOfMeasurement, cls).__new__(cls)

    def __init__(self, symbols: str, /):
        # Set as empty string to skip the coming for-loop
        if symbols is None:
            symbols = ""

        # Record the exponent associated with the units
        self._symbols_powers = dict()

        for symbol in symbols.split(" "):
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

        self._symbols_powers = {k: self._symbols_powers[k] for k in sorted(self._symbols_powers, key=symbol_sorter)}
        return

    def __repr__(self):
        units, = self.args
        return units

    def to_si(self, values):
        """Convert values in the instance's units of measure, to the equivalent values in SI equivalent base units."""
        values = np.atleast_1d(values)

        # Map imperial temperatures, if appropriate
        if self.args[0] == "degF":
            values = values - 32

        # In general
        for (symbol, power) in self._symbols_powers.items():
            transfer_func = self._memo_conv[symbol].get("to_si")
            if not callable(transfer_func):
                raise NotImplementedError("Cannot convert unit to SI equivalent")
            values = transfer_func(values, power)

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
            transfer_func = self._memo_conv[symbol].get("from_si")
            if not callable(transfer_func):
                raise NotImplementedError("Cannot convert unit from SI equivalent")
            values = transfer_func(values, power)

        # Map imperial temperatures, if appropriate
        if self.args[0] == "degF":
            values = values + 32

        return values

    @property
    def args(self) -> tuple[str]:
        """An args tuple that could be used to instantiate a new object with identical units."""
        instantiable_string = " ".join([
            f"{symbol}" if power == 1 else f"{symbol}^{power}"
            for (symbol, power) in self._symbols_powers.items()
        ])
        return (instantiable_string,)

    @property
    def units_si(self) -> str:
        dim_powers = np.vstack([self._memo_dims[k] * v for (k, v) in self._symbols_powers.items()]).sum(axis=0)
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

        # From the units, determine the SI equivalent quantity representation
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
        if results and isinstance(results[0], Quantity):
            results[0]._carpy_units = self._carpy_units

        return results[0] if len(results) == 1 else results

    # ==================================
    # Mathematical built-ins overloading
    # ----------------------------------

    def __abs__(self):
        """Absolute value."""
        return

    def __add__(self, other):
        """Addition."""
        return

    def __ceil__(self):
        """Ceiling function."""
        return

    def __divmod__(self, other):
        """Division quotient, remainder."""
        return

    def __eq__(self, other):
        """Equality."""
        return

    def __float__(self):
        """Cast as float."""
        return

    def __floor__(self):
        """Floor function."""
        return

    def __floordiv__(self, other):
        """self // other, floored division quotient."""
        return

    def __ge__(self, other):
        """Greater than or equal to."""
        return

    def __gt__(self, other):
        """Greater than."""
        return

    def __int__(self):
        """Cast as integer."""
        return

    def __le__(self, other):
        """Less than or equal to."""
        return

    def __lt__(self, other):
        """Less than."""
        return

    def __mod__(self, other):
        """Modulo."""
        return

    def __mul__(self, other):
        """Multiplication."""
        return

    def __neg__(self):
        """Negation"""
        return

    def __pow__(self, power):
        """self raised to a power"""
        return

    def __radd__(self, other):
        """Reverse addition."""
        return self.__add__(other)

    def __rdivmod__(self, other):
        """Reverse divmod."""
        return self.__divmod__(other)

    def __rfloordiv__(self, other):
        """Reverse floor division."""
        return self.__floordiv__(other)

    def __rmod__(self, other):
        """Reverse modulo."""
        return self.__rmod__(other)

    def __rmul__(self, other):
        """Reverse multiplication."""
        return self.__mul__(other)

    def __round__(self, n=None):
        """Round function."""
        return

    def __rpow__(self, other):
        """Reverse raise power."""
        return self.__pow__(other)

    def __rsub__(self, other):
        """Reverse subtraction."""
        return self.__sub__(other)

    def __rtruediv__(self, other):
        """Reverse true division."""
        return self.__truediv__(other)

    def __sub__(self, other):
        """Subtraction."""
        return

    def __truediv__(self, other):
        """True division."""
        return

    def __trunc__(self):
        """Truncate."""
        return

    # =================================================
    # Other methods not directly interacting with numpy
    # -------------------------------------------------

    def __repr__(self):
        """For maths purposes, display SI."""
        repr1 = super(Quantity, self).__repr__()
        repr2 = self._carpy_units.units_si
        return f"{repr1.rstrip()[:-1]}, {repr2})"

    def __str__(self):
        """For display purposes, display whatever the original unit was."""
        rtn_str1 = self._carpy_units.to_uom(self.x).__str__()
        rtn_str2, = self._carpy_units.args
        return f"{rtn_str1} {rtn_str2}"

    def to(self, units: str) -> np.ndarray:
        """Return a numpy array in the chosen units."""
        new_uom = UnitOfMeasurement(units)
        new_value = new_uom.to_uom(self.x)
        return new_value

    @property
    def x(self) -> np.ndarray:
        """Return the values of the internal numpy array without the Quantity class wrapper."""
        return np.array(self)
