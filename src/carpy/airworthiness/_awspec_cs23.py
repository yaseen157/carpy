from warnings import warn
from ._awspec import AWspec

__all__ = ["CS23"]


class CS23(AWspec):
    _amendments = list(range(6))
    _categories = ["normal", "utility", "aerobatic", "commuter"]
    _type = 23

    # SUBPART A -- GENERAL
    @property
    def dot_1(self):
        """CS 23.1 - Applicability."""
        MTOW = self._concept.weights.W_AUW
        npassengers = self._concept.occupants

        if self.category in ["normal", "utility", "aerobatic"]:
            if MTOW <= 5_670 and npassengers <= 9:
                return True
        else:
            if MTOW <= 8_618 and npassengers <= 19:
                warnmsg = "23.1(a)(2): Couldn't check propulsion system type"
                warn(warnmsg, RuntimeWarning)
                return True

        return False
