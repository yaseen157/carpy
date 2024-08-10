from warnings import warn

from carpy.airworthiness import AWspec

from ._occupants import Occupants
from ._speeds import VspeedsFW
from ._weights import Weights


class FixedWingConcept:
    _AWspec = None
    _Vspeeds: VspeedsFW
    _occupants: Occupants
    _weights: Weights

    def __init__(self):
        self._Vspeeds = VspeedsFW(self)
        self._occupants = Occupants(self)
        self._weights = Weights(self)

    @property
    def AWspec(self) -> AWspec:
        """Airworthiness/designed type."""
        return self._AWspec

    @AWspec.setter
    def AWspec(self, value):

        if issubclass(value, AWspec):  # Correct use of class
            awspec_class = value

        elif isinstance(value, AWspec):  # Incorrect use of instance
            warnmsg = (
                f"Do not set '{self.AWspec.__name__}' using an instance, use "
                f"the type of the {AWspec.__name__} e.g. type({value})"
            )
            warn(warnmsg, RuntimeWarning)
            awspec_class = type(value)

        else:
            errormsg = f"Unsupported type '{value}'"
            raise ValueError(errormsg)

        # Instantiate
        self._AWspec = awspec_class(self)

    @property
    def Vspeeds(self) -> VspeedsFW:
        return self._Vspeeds

    @property
    def occupants(self) -> Occupants:
        """Object describing the number of occupants."""
        return self._occupants

    @property
    def weights(self):
        return self._weights
