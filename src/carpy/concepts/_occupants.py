from carpy.utility import NumberSets

__all__ = ["Occupants"]


class Occupants:
    _flightcrew = 1
    _cabincrew = 0
    _passengers = 0

    def __init__(self, __concept, /):
        self.__concept = __concept

    @property
    def _concept(self):
        """Parent vehicle concept."""
        return self.__concept

    @property
    def flightcrew(self) -> int:
        """The number of flight crew operating the aircraft in flight."""
        return self._flightcrew

    @flightcrew.setter
    def flightcrew(self, value):
        self._flightcrew = NumberSets.cast_N(value, safe=True)

    @property
    def cabincrew(self) -> int:
        """The number of cabin crew tending to passengers."""
        return self._cabincrew

    @cabincrew.setter
    def cabincrew(self, value):
        self._cabincrew = NumberSets.cast_N(value, safe=True)

    @property
    def passengers(self) -> int:
        """The number of seated passengers."""
        return self._passengers

    @passengers.setter
    def passengers(self, value):
        self._passengers = NumberSets.cast_N(value, safe=True)
