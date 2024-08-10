"""
A module implementing several diurnal cycles for climatic elements, as prescribed in the MIL-HDBK-310 tables.

References:
    MIL-HDBK-310 Global Climatic Data for Developing Military Products

"""
import os

import numpy as np
import pandas as pd

from carpy.environment.daycycles import DiurnalCycle
from carpy.utility import PathAnchor, Quantity

__all__ = ["MH310hot", "MH310basic", "MH310cold", "MH310coast"]
__author__ = "Yaseen Reza"

# Load the handbook data
anchor = PathAnchor()
filename = "MIL-HDBK-310_diurnals.xlsx"
filepath = os.path.join(anchor.directory_path, "..", "data", filename)
dataframes = pd.read_excel(filepath, sheet_name=None)


def cycle_factory(dataframe: pd.DataFrame):
    local_solar_time = dataframe["Time [LST]"]
    local_temperature = dataframe["Temperature, T [degC]"]
    local_irradiance = dataframe["Solar Irradiance, H0 [W m^{-2}]"]

    # relative_humidity = df["Relative Humidity, RH [%]"]

    def _temperature(self, lst_hrs):
        temperature = np.interp(lst_hrs, local_solar_time, local_temperature)
        return Quantity(temperature, "degC")

    def _irradiance(self, lst_hrs):
        irradiance = np.interp(lst_hrs, local_solar_time, local_irradiance)
        return Quantity(irradiance, "W m^{-2}")

    cycle_class = type("MIL310DayCycle", (DiurnalCycle,), {
        "_temperature": _temperature,
        "_irradiance": _irradiance
    })
    cycle_instance = cycle_class()

    return cycle_instance


class MH310hot:
    """
    MIL-HDBK-310 diurnal cycles for the "hot" regional type.

    Paraphrased from the handbook:
    "This regional type includes the hot subtropical deserts of the world where the 1-percent temperature during the
    worst month exceeds 43.3°C (110°F). The hottest parts of this regional type are the hottest areas of the world.
    Other parts of this regional type, while not as hot, are prone to occurrences of the highest absolute humidities in
    the world. Therefore, two cycles are provided to describe these diverse climatic conditions."
    """

    @staticmethod
    def hot_dry() -> DiurnalCycle:
        """
        Returns:
            Hot and dry diurnal cycle model, for the hot regional type.

        """
        return cycle_factory(dataframe=dataframes["hot_hot-dry"])

    @staticmethod
    def hot_humid() -> DiurnalCycle:
        """
        Returns:
            Hot and humid diurnal cycle model, for the hot regional type.

        """
        return cycle_factory(dataframe=dataframes["hot_hot-dry"])


class MH310basic:
    """
    MIL-HDBK-310 diurnal cycles for the "basic" regional type.

    Paraphrased from the handbook:
    "The basic regional type includes all the land areas of the world that have neither extremely high nor extremely low
    temperatures, as defined above. The basic type incorporates most of the mid-latitudes, which are often referred to
    climatically as temperate, moderate, or intermediate zones. It also includes the humid tropics, which are warm
    throughout the year, but do not record the extremely high temperatures that occur in the hot regional type. The
    basic type is roughly coincident with the more densely populated, industrialized, and agriculturally productive
    areas of the world; therefore, most of the land areas with the highest probability of combat operations are within
    its limits.

    In addition to the basic hot and basic cold temperature cycles that delineate this regional type, three daily
    weather cycles are included because of their high humidity conditions. These are: (1) constant high humidity;
    (2) variable high humidity; and (3) cold-wet. The first two are conditions associated with the humid tropics, where
    they occur with high frequency throughout the year. They also occur in parts of the subtropics and mid latitudes
    during the summer. The cold-wet cycle occurs in certain mid latitude areas during winter, and is characterized by
    frequent temperatures near 0°C, with concurrent high relative humidity and frequent precipitation, including frozen
    varieties."

    A detailed description for each cycle has been lifted from the handbook and provided in the appropriate docstring.
    """

    @staticmethod
    def humid_variable() -> DiurnalCycle:
        """
        The variable high humidity cycle occurs in the tropics in open areas with clear skies or intermittent
        cloudiness, with consequent daily control of temperature and humidity by the solar radiation cycle. Items will
        be subject to alternate wetting and drying.

        Returns:
            Variable high humidity diurnal cycle model, for the basic regional type.

        """
        return cycle_factory(dataframe=dataframes["basic_humid-variable"])

    @staticmethod
    def humid_constant() -> DiurnalCycle:
        """
        The constant high humidity cycle is the result of conditions in heavily forested areas in the tropics under
        thick cloud cover, which tends to produce near constancy of temperature, solar radiation, and humidity near the
        ground during rainy seasons. Exposed materiel is likely to be constantly wet or damp for many days at a time.

        Returns:
            Constant high humidity diurnal cycle model, for the basic regional type.

        """
        return cycle_factory(dataframe=dataframes["basic_humid-constant"])

    @staticmethod
    def hot() -> DiurnalCycle:
        """
        These conditions occur in sections of the United States, Mexico, northern Africa, southwestern Asia, India,
        Pakistan, and southern Spain in the Northern Hemisphere, and smaller sections of South America, southern Africa,
        and Australia in the Southern Hemisphere.

        Returns:
            Hot diurnal cycle model, for the basic regional type.

        """
        return cycle_factory(dataframe=dataframes["basic_hot"])

    @staticmethod
    def cold() -> DiurnalCycle:
        """
        Extensive basic cold areas occur only in the Northern Hemisphere, in the northern United States, the coast of
        Alaska, southern Canada, the coast of southern Greenland, northern Europe, the Soviet Union, and Central Asia.
        Small, isolated areas of basic cold conditions may be found at high elevations in lower latitudes.

        Returns:
            Cold diurnal cycle model, for the basic regional type.

        """
        return cycle_factory(dataframe=dataframes["basic_cold"])

    @staticmethod
    def cold_wet() -> DiurnalCycle:
        """
        Basic cold-wet conditions occur throughout the colder, humid sections of the basic regional type adjoining the
        areas of basic cold conditions. Cold-wet conditions, as defined here, may occur in any part of the basic type
        that regularly experiences freezing and thawing on a given day; however, the conditions are found most
        frequently in Western Europe, the central United States, and northeastern Asia (China and Japan). In the
        Southern Hemisphere, cold-wet conditions occur only at moderately high elevations except in South America where
        they are found in Argentina and Chile south of 40º latitude.

        Returns:
            Cold-wet diurnal cycle model, for the basic regional type.

        """
        return cycle_factory(dataframe=dataframes["basic_cold-wet"])


class MH310cold:
    """
    MIL-HDBK-310 diurnal cycles for the "cold" regional type.

    Paraphrased from the handbook:
    "The cold regional type is characterized by temperatures that are lower than those of the basic cold daily weather
    cycle, but higher than those of the severe cold regional type. The cold regional type requires only one daily
    weather cycle to establish its range of conditions. Conditions are found in areas including most of Canada, and
    large sections of Alaska, Greenland, northern Scandinavia, northeastern USSR, and Mongolia. Cold conditions also
    exist in parts of the Tibetan Plateau of Central Asia and at high elevations in both the Northern and Southern
    Hemispheres."
    """

    @staticmethod
    def cold() -> DiurnalCycle:
        """
        Returns:
            Cold diurnal cycle model, for the cold regional type.

        """
        return cycle_factory(dataframe=dataframes["cold_cold"])


class MH310coast:
    """
    MIL-HDBK-310 diurnal cycles for the "coastal" regional type.

    Paraphrased from the handbook:
    "This regional type includes open seas and coastal ports north of 60°S. Climatic data are excluded for the periods
    during which locations are closed to navigation due to sea ice.

    In general, equipment should be designed to operate during all but a small percentage of the time. More extreme
    climatic values should be considered for equipment whose failure to operate is life-threatening, or for materiel
    that could be rendered useless or dangerous after one-time exposure. Another option for such material would be
    protection from exposure to these extremes."

    A detailed description for each cycle has been lifted from the handbook and provided in the appropriate docstring.
    """

    @staticmethod
    def hot() -> DiurnalCycle:
        """
        Daily cycle of temperature and other elements associated with the 1% high temperature value for the
        coastal/ocean regional type.

        Returns:
            Hot diurnal cycle model, for the coastal/ocean regional type.

        """
        return cycle_factory(dataframe=dataframes["coastal_hot"])

    @staticmethod
    def hot_humid() -> DiurnalCycle:
        """
        Daily cycle of relative humidity and temperature (including solar radiation) associated with the 1% high
        relative humidity with high temperature value for the coastal/ocean regional type.

        Returns:
            Hot-humid diurnal cycle model, for the costal/ocean regional type.

        """
        return cycle_factory(dataframe=dataframes["coastal_hot-humid"])

    @staticmethod
    def hot_dry() -> DiurnalCycle:
        """
        Daily cycle of relative humidity and temperature (including solar radiation) associated with the 1% low relative
        humidity with high temperature value for the coastal/ocean regional type.

        Returns:
            Hot-dry diurnal cycle model, for the costal/ocean regional type.

        """
        return cycle_factory(dataframe=dataframes["coastal_hot-dry"])
