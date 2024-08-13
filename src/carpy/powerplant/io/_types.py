"""Module defining the types of inputs and outputs to powerplant network modules."""
import warnings

import numpy as np

from carpy.utility import Quantity

__all__ = ["AbstractPower", "Chemical", "Electrical", "Mechanical", "Thermal", "Radiant", "Fluid"]
__author__ = "Yaseen Reza"


class AbstractPower:
    """Base class for describing types of power that can be input or output of a powerplant module."""
    _power = Quantity(np.nan, "W")

    def __init__(self, power):
        self.power = power
        return

    def __repr__(self):
        repr_str = f"<{AbstractPower.__name__} '{type(self).__name__}' @ {hex(id(self))}>"
        return repr_str

    def __irshift__(self, other):
        """For compatibility with plant modules... not to be used otherwise"""
        error_msg = (f"'{type(self).__name__}' power type cannot be declared as an input to any power plant modules. "
                     f"Instead, use power types to specify the required output(s) of a plant's network")
        raise AttributeError(error_msg)

    @property
    def power(self) -> Quantity:
        """Real or useful power."""
        return self._power

    @power.setter
    def power(self, value):
        self._power = Quantity(value, "W")

    def reverse_pass(self):
        return


class Chemical(AbstractPower):
    """
    Chemical power, defined by calorific value and mass flow rate.

    This type is fully defined when 'power' and 'mdot' attributes are set.
    """
    _mdot = Quantity(0, "kg s^-1")

    @property
    def mdot(self) -> Quantity:
        """Mass flow rate."""
        return self._mdot

    @mdot.setter
    def mdot(self, value):
        self._mdot = Quantity(value, "kg s^-1")

    @property
    def CV(self) -> Quantity:
        """Calorific value, i.e. gravimetric energy density."""
        return self.power / self.mdot

    @CV.setter
    def CV(self, value):
        self.mdot = self.power / value


class Electrical(AbstractPower):
    """
    Sinusoidal electrical power. Set frequency omega to zero for DC modelling.

    This type is fully defined when 'power', 'V_rms', 'X', and 'omega' attributes are set.
    """
    _V_rms = Quantity(np.nan, "V")
    _X = Quantity(0, "ohm")
    _omega = Quantity(0, "Hz")

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
    def C(self) -> Quantity:
        """Equivalent electrical capacitance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore the divide by zero error from direct current
            return 1 / self.X / self.omega

    @C.setter
    def C(self, value):
        self.X = 1 / value / self.omega

    @property
    def L(self) -> Quantity:
        """Equivalent electrical inductance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore the divide by zero error from direct current
            return self.X / self.omega

    @L.setter
    def L(self, value):
        self.X = value * self.omega

    @property
    def power_factor(self):
        """Circuit power factor."""
        pf = np.cos(self.phi)
        return pf

    @power_factor.setter
    def power_factor(self, value):
        self.phi = np.arccos(value)

    @property
    def Q(self) -> Quantity:
        """Reactive power. A negative reactive power lags the active power by 90 degrees."""
        return Quantity(self.power * np.tan(self.phi), "var")

    @Q.setter
    def Q(self, value):
        self.phi = np.arctan2(value, self.power)

    @property
    def R(self) -> Quantity:
        """Electrical resistance."""
        return self.V_rms / self.I_rms

    @R.setter
    def R(self, value):
        self.I_rms = (self.power / Quantity(value, "ohm")) ** 0.5

    @property
    def S(self) -> Quantity:
        """Complex power."""
        return Quantity(self.power + self.Q * 1j, "VA")

    @S.setter
    def S(self, value):
        S = Quantity(value, "VA")
        self.phi = np.arctan2(S.imag, S.real)

    @property
    def S_mag(self) -> Quantity:
        """Apparent power."""
        return Quantity(np.abs(self.S), "VA")

    @S_mag.setter
    def S_mag(self, value):
        Q_mag = (self.power ** 2 - value ** 2) ** 0.5
        if self.X >= 0:
            self.Q = Q_mag
        else:
            self.Q = -Q_mag

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
    def phi(self):
        """Phase of voltage relative to current (phi = arg(V) - arg(I))."""
        return np.arctan2(self.X, self.R)

    @phi.setter
    def phi(self, value):
        self.X = self.R * np.tan(value)


class Mechanical(AbstractPower):
    def __init__(self):
        raise NotImplementedError


class Thermal(AbstractPower):
    def __init__(self):
        raise NotImplementedError


class Radiant(AbstractPower):
    """
    Electromagnetical power, i.e. power transferred via electromagnetic irradiance.
    """
    pass


class Fluid(AbstractPower):
    """
    Fluidal power, i.e. the product of pressure and volumetric flow.

    This type is fully defined when 'power', 'mdot', and 'pressure' attributes are set.
    """
    _mdot = Quantity(0, "kg s^-1")
    _pressure = Quantity(np.nan, "Pa")

    @property
    def mdot(self) -> Quantity:
        """Mass flow rate."""
        return self._mdot

    @mdot.setter
    def mdot(self, value):
        self._mdot = Quantity(value, "kg s^-1")

    @property
    def pressure(self) -> Quantity:
        """Fluid pressure."""
        return self._pressure

    @pressure.setter
    def pressure(self, value):
        self._pressure = Quantity(value, "Pa")

    @property
    def Vdot(self) -> Quantity:
        """Volumetric flow rate."""
        return self.power / self.pressure

    @Vdot.setter
    def Vdot(self, value):
        self.pressure = self.power / value

    @property
    def rho(self) -> Quantity:
        """Fluid density."""
        return self.mdot / self.Vdot

    @rho.setter
    def rho(self, value):
        self.Vdot = self.mdot / value
