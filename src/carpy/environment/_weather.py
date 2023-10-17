"""Module implementing means of accessing and parsing weather reports."""
import time
import warnings

from metar.Metar import Metar
import numpy as np
import requests

from carpy.utility import call_count

__all__ = ["METARtools"]
__author__ = "Yaseen Reza"


@call_count
def _metar_from_api(*station_ids: str, hours: int = None) -> str:
    """
    Return a string of historical METAR observations at desired stations. This
    is achieved by querying 'https://aviationweather.gov'. Calls to this
    function are not rate limited, so please use responsibly (ideally not
    at all if you are an end-user).

    Args:
        *station_ids: String arguments of ICAO station IDs to query
        hours: Integer number of hours from which to pull historical METAR
            records. Optional, defaults to 3 hour.

    Returns:
        METAR string with reports separated by the newline character.

    Examples: ::

        # Get last 2 hrs of data for London Heathrow and Southampton Airport
        >>> mymetars = _metar_from_api("EGLL", "EGHI", hours=2)

        # Read the first line of the metar string
        >>> print(mymetars.splitlines()[0])
        'EGHI 270850Z 12005G15KT 090V190 9999 BKN025 11/06 Q1015'

    """
    # Recast as necessary
    hours = int(3 if hours is None else hours)

    # Build request
    url_base = "https://aviationweather.gov/"
    api_path = "cgi-bin/data/metar.php?ids="
    api_querytext = ",".join(station_ids) + f"&hours={hours}"

    # Perform query
    try:
        link = f"{url_base}{api_path}{api_querytext}"
        response = requests.get(url=link)
    except Exception as _:
        errormsg = "Couldn't access weather service. Is your device connected?"
        raise ConnectionError(errormsg)

    if response.status_code != 200:
        errormsg = f"Couldn't access METAR data ({response.status_code=})"
        raise ConnectionError(errormsg)

    # Return results of a successful query
    return response.text


class METARtools(object):
    """A collection of tools for working with METAR observations."""

    @staticmethod
    def fetch(*station_ids: str, hours: int = None) -> list:
        """
        Return a string of historical METAR observations at desired stations.
        Calls to the API embedded in this method are rate limited to prevent
        and discourage spam to the API host.

        Args:
            *station_ids: String arguments of ICAO station IDs to query
            hours: Integer number of hours from which to pull historical METAR
                records. Optional, defaults to 3 hour.

        Returns:
            METAR string with reports separated by the newline character.

        Examples: ::

            # Get last 2 hrs of data for London Heathrow and Southampton Airport
            >>> mymetars = METARtools.fetch("EGLL", "EGHI", hours=2)

            # Read the first line of the metar string
            >>> print(mymetars[0])
            'EGHI 270850Z 12005G15KT 090V190 9999 BKN025 11/06 Q1015'

        """
        # Rate-limit calls to the API according to a logarithmic schedule
        api_response = _metar_from_api(*station_ids, hours=hours)
        api_calls = _metar_from_api.call_count
        time.sleep(np.log(api_calls + 1))

        # Let's be real, most users want a list of METARs, not a single string
        metarlist = api_response.splitlines()
        return metarlist

    @staticmethod
    def decode(*METARs: str) -> list:
        """
        Return decoded METAR observations.

        Args:
            *METARs: String arguments of coded METAR observations.

        Returns:
            A list of Metar objects that corresponding to the argument METARs.

        """
        decoded_obs = [Metar(metar_str) for metar_str in METARs]
        return decoded_obs
