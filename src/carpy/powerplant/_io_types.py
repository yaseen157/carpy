"""Module defining the types of inputs and outputs to powerplant network modules."""
import warnings

import numpy as np

from carpy.utility import Quantity

__all__ = []
__author__ = "Yaseen Reza"


class IOPower:
    """Base class for describing types of power that can be input or output of a powerplant module."""
    _power = Quantity(np.nan, "W")

    @property
    def power(self) -> Quantity:
        """Real or useful power."""
        return self._power

    @power.setter
    def power(self, value):
        self._power = Quantity(value, "W")


class Chemical(IOPower):
    """Chemical power, defined by calorific value and mass flow rate."""
    _CV = Quantity(np.nan, "J kg^-1")

    @property
    def CV(self) -> Quantity:
        """Calorific value."""
        return self._CV

    @CV.setter
    def CV(self, value):
        self._CV = Quantity(value, "J kg^-1")

    @property
    def mdot(self) -> Quantity:
        """Mass flow rate."""
        return self.power / self.CV

    @mdot.setter
    def mdot(self, value):
        self.CV = self.power / value


class Electrical(IOPower):
    """Sinusoidal electrical power."""
    _S_mag = Quantity(np.nan, "VA")
    _V_rms = Quantity(np.nan, "V")
    _X = Quantity(0, "ohm")
    _omega = Quantity(0, "Hz")

    @property
    def S_mag(self) -> Quantity:
        """Apparent power."""
        return np.abs(self.S)

    @S_mag.setter
    def S_mag(self, value):
        Q_mag = (self.power ** 2 - value ** 2) ** 0.5
        if self.X >= 0:
            self.Q = Q_mag
        else:
            self.Q = -Q_mag

    @property
    def V_rms(self) -> Quantity:
        """Root mean square electrical potential."""
        return self._V_rms

    @V_rms.setter
    def V_rms(self, value):
        self._V_rms = Quantity(value, "V")

    @property
    def omega(self):
        """Signal frequency."""
        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = Quantity(value, "Hz")

    @property
    def X(self) -> Quantity:
        """Electrical reactance."""
        return self._X

    @X.setter
    def X(self, value):
        self._X = Quantity(value, "ohm")

    @property
    def I_rms(self) -> Quantity:
        """Electrical current."""
        return self.power / self.V_rms

    @I_rms.setter
    def I_rms(self, value):
        self.V_rms = self.power / value

    @property
    def phi(self):
        """Phase of voltage relative to current (phi = arg(V) - arg(I))."""
        return np.arctan2(self.X, self.R)

    @phi.setter
    def phi(self, value):
        self.X = self.R * np.tan(value)

    @property
    def R(self) -> Quantity:
        """Electrical resistance."""
        return self.V_rms / self.I_rms

    @R.setter
    def R(self, value):
        self.I_rms = (self.power / Quantity(value, "ohm")) ** 0.5

    @property
    def C(self) -> Quantity:
        """Electrical capacitance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore the divide by zero error from direct current
            return 1 / self.X / self.omega

    @C.setter
    def C(self, value):
        self.X = 1 / value / self.omega

    @property
    def L(self) -> Quantity:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore the divide by zero error from direct current
            return self.X / self.omega

    @L.setter
    def L(self, value):
        self.X = value * self.omega

    @property
    def S(self) -> Quantity:
        """Complex power."""
        return Quantity(self.power + self.Q * 1j, "VA")

    @S.setter
    def S(self, value):
        S = Quantity(value, "VA")
        self.phi = np.arctan2(S.imag, S.real)

    @property
    def Q(self) -> Quantity:
        """Reactive power."""
        return self.power * np.tan(self.phi)

    @Q.setter
    def Q(self, value):
        self.phi = np.arctan2(value, self.power)

    @property
    def Z(self) -> Quantity:
        """Electrical impedance."""
        return Quantity(self.R + self.X * 1j, "ohm")

    @Z.setter
    def Z(self, value):
        Z = Quantity(value, "ohm")
        self.R = Z.real
        self.X = Z.imag

    @property
    def power_factor(self):
        """Circuit power factor."""
        pf = np.cos(self.phi)
        return pf

    @power_factor.setter
    def power_factor(self, value):
        self.phi = np.arccos(value)
