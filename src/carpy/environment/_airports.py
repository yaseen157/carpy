"""Module for accessing properties of airports and their runways."""
import os
from functools import wraps
import warnings

import numpy as np
import pandas as pd

from carpy.utility import GetPath, Quantity, isNone

__all__ = ["Airport", "Runway"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Load data required by the module
# ---------------------------------------------------------------------------- #


def load_ourairports(filename: str, /):
    """
    Load and return an offline image of data relating to airfields.

    Args:
        filename: The filename and extension to be loaded.

    References:
        https://ourairports.com/
        https://ourairports.com/help/data-dictionary.html

    Notes:
        Data is offline (imaged), and so may not be up-to-date!.

    """
    # Load the actual data
    oa_data = os.path.join(GetPath.localpackage(), "data", "ourairports")
    filepath = os.path.join(oa_data, filename)
    dataframe = pd.read_csv(filepath_or_buffer=filepath)

    # Load the definitions (if available)
    try:
        sheetname, _ = filename.rsplit(".")
        filepath = os.path.join(oa_data, "definitions.xlsx")
        definitions = pd.read_excel(io=filepath, sheet_name=sheetname)
    except Exception as _:
        errormsg = (
            f"Could not find definitions for 'ourairports' data: {filename}"
        )
        raise ValueError(errormsg)

    # Package the data and definitions
    classargs = (
        f"{sheetname}data",
        (object,),
        dict([("df", dataframe), ("defs", definitions)])
    )
    dataobject = type(*classargs)

    return dataobject


oa_airports = load_ourairports("airports.csv")
oa_runways = load_ourairports("runways.csv")
warnings.warn(
    message="Using local image of OurAirports data, dated 22/02/2023.",
    category=RuntimeWarning
)


# ============================================================================ #
# Support functions
# ---------------------------------------------------------------------------- #

def idx0(func):
    """
    Decorator for returning the first element of a series returned by the
    wrapped function.

    Args:
        func: Function returning a Pandas series.

    Returns:
        The first element of the series returned by the wrapped function.

    Notes:
        Can also find the first index of an array, but this use case isn't
            officially supported for users - internal use only at this time.

    """
    if callable(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Function wrapper giving 1st element in a pandas series obj."""
            result = func(*args, **kwargs)
            if isinstance(result, pd.Series):
                return result.iloc[0]
            else:
                return result[0]

        return wrapper

    # Technically the below are exceptions for unexpected uses...
    elif isinstance(func, pd.Series):
        # Oops, not a function returning a series. It's an actual series!
        return func.iloc[0]

    else:
        # Who knows. Just index 0 and hope for the best
        return func[0]


# ============================================================================ #
# Airfield objects
# ---------------------------------------------------------------------------- #


class Runway(object):
    """
    Runway objects describe properties of a runway surface including the
    runway's names (in both directions), position of each runway end, and
    dimensions.
    """

    def __init__(self, ourairports_id: int):
        """
        Args:
            ourairports_id: Numerical ID used by OurAirports internal database.
        """
        self._df = oa_runways.df.loc[oa_runways.df["id"] == ourairports_id]
        return

    def __repr__(self):
        my_id = idx0(self._df["id"])
        return f"{type(self).__name__}(ourairports_id={my_id})"

    def __str__(self):
        return f"RWY {self.name}"

    @property
    @idx0
    def airport_ident(self) -> str:
        """
        The text identifier used in the OurAirports URL of the parent airport.
        This will be the ICAO code if available. Otherwise, it will be a local
        airport code (if no conflict), or if nothing else is available, an
        internally-generated code starting with the ISO2 country code, followed
        by a dash and a four-digit number.
        """
        return self._df["airport_ident"]

    @property
    def name(self) -> str:
        """
        The name of the runway.

        Returns:
            <low-end identifier>/<high-end identifier>

        """
        return f"{idx0(self._df['le_ident'])}/{idx0(self._df['he_ident'])}"

    @property
    def length(self) -> Quantity:
        """
        Length of the full runway surface (including displaced thresholds,
        overrun areas, etc.).
        """
        # Convert ourairports data (in feet) to Quantity objects
        return Quantity(idx0(self._df["length_ft"]), "ft")

    @property
    def width(self) -> Quantity:
        """Width of the runway surface."""
        # Convert ourairports data (in feet) to Quantity objects
        return Quantity(idx0(self._df["width_ft"]), "ft")

    @property
    @idx0
    def _surface(self) -> str:
        """Runway surface type."""
        warnmsg = (
            "The '._surface' attribute of runways doesn't adhere to a "
            "controlled vocabulary, therefore code utilising this attribute "
            "should be built and used with caution"
        )
        warnings.warn(message=warnmsg, category=RuntimeWarning)
        return self._df["surface"]

    @property
    def lighted(self) -> bool:
        """True if the surface is lighted at night, else False."""
        return bool(idx0(self._df["lighted"]))

    @property
    def closed(self) -> bool:
        """True if the runway surface is closed, else False."""
        return bool(idx0(self._df["closed"]))

    @property
    @idx0
    def le_ident(self) -> str:
        """Identifier for the low-numbered end of the runway."""
        return self._df["le_ident"]

    @property
    @idx0
    def le_latitude_deg(self) -> float:
        """
        Latitude of the centre of the low-numbered end of the runway, in
        decimal degrees (+ve is north), if available.
        """
        return self._df["le_latitude_deg"]

    @property
    @idx0
    def le_longitude_deg(self) -> float:
        """
        Longitude of the centre of the low-numbered end of the runway, in
        decimal degrees (+ve is east), if available."""
        return self._df["le_longitude_deg"]

    @property
    def le_elevation(self) -> Quantity:
        """
        Elevation above mean sea level of the low-numbered end of the runway.
        """
        return Quantity(idx0(self._df["le_elevation_ft"]), "ft")

    @property
    @idx0
    def le_heading_degT(self) -> float:
        """
        Heading of the low-numbered end of the runway in degrees true (not
        magnetic).
        """
        return self._df["le_heading_degT"]

    @property
    def le_displaced_threshold(self) -> Quantity:
        """
        Length of the displaced threshold (if any) for the low-numbered end of
        the runway. Displaced thresholds may be used for taxiing, takeoff, and
        landing rollout, but not for touchdown.
        """
        displacement = idx0(self._df["le_displaced_threshold_ft"])

        # If there is no displacement, take default value of zero (displacement)
        if np.isnan(displacement):
            displacement = 0

        return Quantity(displacement, "ft")

    @property
    @idx0
    def he_ident(self) -> str:
        """Identifier for the high-numbered end of the runway."""
        return self._df["he_ident"]

    @property
    @idx0
    def he_latitude_deg(self) -> float:
        """
        Latitude of the centre of the high-numbered end of the runway, in
        decimal degrees (+ve is north), if available.
        """
        return self._df["he_latitude_deg"]

    @property
    @idx0
    def he_longitude_deg(self) -> float:
        """
        Longitude of the centre of the high-numbered end of the runway, in
        decimal degrees (+ve is east), if available."""
        return self._df["he_longitude_deg"]

    @property
    def he_elevation(self) -> Quantity:
        """
        Elevation above mean sea level of the high-numbered end of the runway.
        """
        return Quantity(idx0(self._df["he_elevation_ft"]), "ft")

    @property
    @idx0
    def he_heading_degT(self) -> float:
        """
        Heading of the high-numbered end of the runway in degrees true (not
        magnetic).
        """
        return self._df["he_heading_degT"]

    @property
    def he_displaced_threshold(self) -> Quantity:
        """
        Length of the displaced threshold (if any) for the high-numbered end of
        the runway. Displaced thresholds may be used for taxiing, takeoff, and
        landing rollout, but not for touchdown.
        """
        displacement = idx0(self._df["he_displaced_threshold_ft"])

        # If there is no displacement, take default value of zero (displacement)
        if np.isnan(displacement):
            displacement = 0

        return Quantity(displacement, "ft")


class Airport(object):
    """
    Airport objects are used to quickly access size, position, and orientation
    of available runways.
    """

    def __init__(self, ident: str = None, ourairports_id: int = None):
        """
        Args:
            ident: The text identifier used in the OurAirports URL. This will be
                the ICAO code if available. Otherwise, it will be a local
                airport code (if no conflict), or if nothing else is available,
                an internally-generated (by OurAirports) code starting with the
                ISO2 country code, followed by a dash and a four-digit number.
            ourairports_id: Numerical ID used by OurAirports internal database.
        """
        # Sanitise input
        if all(isNone(ident, ourairports_id)):
            errormsg = (
                f"Missing arguments, please supply one of 'ident' or "
                f"'ourairports_id'."
            )
            raise ValueError(errormsg)
        elif ident and ourairports_id:
            errormsg = (
                f"Too many arguments, please supply only one of 'ident' or "
                f"'ourairports_id'."
            )
            raise ValueError(errormsg)

        # Uniquely ID the airport
        if ident:
            df = oa_airports.df.loc[oa_airports.df["ident"] == ident]
        else:
            df = oa_airports.df.loc[oa_airports.df["id"] == ourairports_id]
            ident = df["ident"]
        self._df = df

        # List unique runway surfaces
        rwys_raw = oa_runways.df.loc[oa_runways.df["airport_ident"] == ident]
        self._runways = [Runway(row["id"]) for _, row in rwys_raw.iterrows()]
        return

    def __repr__(self):
        return f"{type(self).__name__}(ident={self.ident})"

    def __str__(self):
        return f"{self.name}"

    @property
    @idx0
    def ident(self) -> str:
        """
        The text identifier used in the OurAirports URL. This will be the ICAO
        code if available. Otherwise, it will be a local airport code (if no
        conflict), or if nothing else is available, an internally-generated code
        starting with the ISO2 country code, followed by a dash and a four-digit
        number.
        """
        return self._df["ident"]

    @property
    @idx0
    def name(self) -> str:
        """The official airport name."""
        return self._df["name"]

    @property
    @idx0
    def latitude_deg(self) -> float:
        """The airport latitude in decimal degrees (+ve for north)."""
        return self._df["latitude_deg"]

    @property
    @idx0
    def longitude_deg(self) -> float:
        """The airport longitude in decimal degrees (+ve for east)."""
        return self._df["longitude_deg"]

    @property
    def elevation(self) -> Quantity:
        """The airport elevation above mean sea level."""
        # Convert ourairports data (in feet) to Quantity objects
        newobject = Quantity(idx0(self._df["elevation_ft"]), "ft")
        return newobject

    @property
    def runways(self) -> list:
        """A list of runway surfaces present at the airport."""
        return self._runways
