"""Module implementing the basis class for modelling diurnal cycles."""
import datetime as dt
import re
import warnings

from carpy.utility import Quantity

__all__ = ["DiurnalCycle"]
__author__ = "Yaseen Reza"


def timecode_to_datetime(value: str | dt.time) -> dt.time:
    """For a string or datetime object argument, use ISO 8601 timecode string as a datetime.time object."""
    # If we started with a datetime.time object, assume it was a solar offset and set it straight away
    if isinstance(value, dt.time):
        return dt.time(
            hour=value.hour,
            minute=value.minute,
            second=value.second,
            microsecond=value.microsecond
        )
    elif value is None:
        return dt.time()
    elif isinstance(value, (float, int)):
        error_msg = (
            f"Use of integers and floats to describe time of day is ambiguous. Use a datetime.time object, or "
            f"ISO 8601 format (e.g. '08:30:48' refers to 8 hours, 30 minutes, and 48 seconds local time)"
        )
        raise ValueError(error_msg)

    # If the setting value is a string, use ISO 8601 unambiguous (solar time) context to parse the string
    iso8601 = re.compile(r"""
            ^T?     (\d{2})  # Match  hh
            (?: (:?)(\d{2})  # Match :mm
            (?: (:?)(\d{2})  # Match :ss
            (?:  \. (\d{3})  # Match .sss
            )?)?)?$
            """, flags=re.VERBOSE)

    if (match := iso8601.match(value)) is None:
        error_msg = f"Could not parse {value} as a valid ISO 8601 time code"
        raise ValueError(error_msg)

    # A valid ISO 8601 match will unpack like this, but we're not done checking that the timecode was valid
    hh, colon1, mm, colon2, ss, sss = match.groups()

    # The standard allows colons to be provided - but it must be all or nothing (so long as seconds are specified)
    if ss and (colon1 != colon2):
        warn_later = True
    else:
        warn_later = False

    # Convert to numerals and assert validity
    hh, mm, ss, sss = map(lambda x: int(x) if x is not None else 0, (hh, mm, ss, sss))  # Convert to numerals
    if not (hh < 24) or not (0 <= mm < 60) or not (0 <= ss < 60) or not (0 <= sss < 1000):
        error_msg = (
            f"The timecode {value} is not considered valid. Please ensure the format of your timecode is "
            f"hh:mm:ss.sss where hh must be 0..23, mm 0..59, ss 0..59, and sss 0..999"
        )
        raise ValueError(error_msg)

    if warn_later:
        warn_msg = f"{value} is not a valid ISO 8601 timecode, but it was interpreted as {hh}:{mm}:{ss}.{sss}"
        warnings.warn(message=warn_msg, category=RuntimeWarning)

    # Finally, build a datetime.time object from the string's components
    return dt.time(hour=hh, minute=mm, second=ss, microsecond=sss * 1_000)


class DiurnalCycle:
    """Base class for modelling daily changes in conditions."""
    _offset_dt: dt.time
    _oneday: Quantity
    _planet: str = None

    def __init__(self, t_oneday=None, solar_offset: dt.time | str = None):
        """
        Args:
            t_oneday: The length of time in one solar day, measured in Earth seconds. Optional, defaults to the length
                of one Earth day.
            solar_offset: The time datum from which the 't_elapsed' argument in this class' methods counts. Users can
                provide an ISO 8601 compatible timecode, or a datetime.time object. Optional, defaults to 10 am.

        """
        # Define the duration of one dirunal cycle in Earth seconds
        if t_oneday is None:
            self._oneday = Quantity(24, "hr")
            self._planet = "Earth"
        else:
            self._oneday = t_oneday

        # Set the solar offset, if one is present
        if solar_offset is None:
            self.solar_offset = dt.time(hour=10)
        else:
            self.solar_offset = solar_offset

    @property
    def solar_offset(self) -> dt.time:
        """The local solar time (LST) offset of the diurnal cycle from midnight."""
        return self._offset_dt

    @solar_offset.setter
    def solar_offset(self, value: dt.time | str):
        self._offset_dt = timecode_to_datetime(value=value)
        return

    @solar_offset.deleter
    def solar_offset(self):
        self._offset_dt = dt.time(hour=10)
        return

    def solar_hour(self, t_elapsed, solar_offset: dt.time | str = None) -> float:
        """
        Compute and return the local solar time (LST) in hours, given the time elapsed (from the offset datum).

        Args:
            t_elapsed: The number of Earth seconds that have elapsed since the solar_offset datum point.
            solar_offset: The diurnal cycle offset. Optional, uses the instance's offset if one has been set already.

        Returns:
            The local solar time, expressed in hours.

        """
        # Get the solar offset into float number of hours
        solar_offset = self.solar_offset if solar_offset is None else timecode_to_datetime(value=solar_offset)
        hours_offset = (
                solar_offset.hour
                + solar_offset.minute / 60
                + solar_offset.second / 3_600
                + solar_offset.microsecond / 3_600e6
        )
        solar_hour = (hours_offset + t_elapsed / (self._oneday.item() / 24)) % 24
        return solar_hour

    def _irradiance(self, lst_hrs) -> Quantity:
        raise NotImplementedError

    def irradiance(self, t_elapsed, solar_offset: dt.time | str = None) -> Quantity:
        """
        The incident solar radiant flux received per unit area.

        Args:
            t_elapsed: The number of Earth seconds that have elapsed since the solar_offset datum point.
            solar_offset: The diurnal cycle offset. Optional, uses the instance's offset if one has been set already.

        Returns:
            Solar irradiance.

        """
        lst_hrs = self.solar_hour(t_elapsed=t_elapsed, solar_offset=solar_offset)
        return self._irradiance(lst_hrs=lst_hrs)

    def _temperature(self, lst_hrs) -> Quantity:
        raise NotImplementedError

    def temperature(self, t_elapsed, solar_offset: dt.time | str = None) -> Quantity:
        """
        The surface temperature.

        Args:
            t_elapsed: The number of Earth seconds that have elapsed since the solar_offset datum point.
            solar_offset: The diurnal cycle offset. Optional, uses the instance's offset if one has been set already.

        Returns:
            Temperature.

        """
        lst_hrs = self.solar_hour(t_elapsed=t_elapsed, solar_offset=solar_offset)
        return self._temperature(lst_hrs=lst_hrs)

    def radiant_exposure(self):
        raise NotImplementedError
