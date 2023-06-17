from carpy.utility import Quantity

__all__ = ["PhotovoltaicPanel", "Battery", "PSU"]


class PhotovoltaicPanel(object):

    def __init__(self):
        self._P_out = Quantity(200, "W")
        return

    def __repr__(self):
        return "PV"

    def P_out(self, P_in=None, **kwargs):
        return self._P_out


class PSU(object):
    _P_input_type = "electrical"

    def __init__(self, eta):
        self._eta = eta
        return

    def __repr__(self):
        return "PSU"

    def P_out(self, P_in=None, **kwargs):
        return P_in * self._eta


class Battery(object):
    _P_input_type = "electrical"

    def __init__(self, specific_energy=None):
        specific_energy = 80 if specific_energy is None else specific_energy
        self.specific_energy = Quantity(specific_energy, "W h kg^{-1}")
        return

    def __repr__(self):
        return "Battery"