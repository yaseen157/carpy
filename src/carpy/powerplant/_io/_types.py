"""Module defining the types of inputs and outputs to powerplant network modules."""
import warnings

import numpy as np

from carpy.physicalchem import FluidState
from carpy.utility import Quantity

__all__ = ["AbstractPower", "Chemical", "Electrical", "Mechanical", "Thermal", "Radiant", "Fluid", "collect"]
__author__ = "Yaseen Reza"


class AbstractPower:
    """Base class for describing types of power that can be input or output of a power plant module."""
    _power: property

    def __init__(self, **kwargs):

        for (k, v) in kwargs.items():
            if k not in dir(self):
                error_msg = f"'{k}' is not recognised as a valid attribute of {type(self).__name__} objects"
                raise AttributeError(error_msg)
            setattr(self, k, v)
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
        error_msg = (
            f"The 'power' attribute of '{type(self).__name__}' class instances should not be set directly. Consult "
            f"the documentation for a valid list of parameters (from which power can be derived)."
        )
        raise ValueError(error_msg)


class Chemical(AbstractPower):
    """
    Chemical power, defined by calorific value and mass flow rate.

    This type is fully defined when 'CV' and 'mdot' attributes are set.
    """

    @property
    def _power(self):
        return self.CV * self.mdot

    _CV = Quantity(np.nan, "J kg^-1")
    _mdot = Quantity(np.nan, "kg s^-1")

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
        return self._CV

    @CV.setter
    def CV(self, value):
        self._CV = Quantity(value, "J kg^-1")


class Electrical(AbstractPower):
    """
    Sinusoidal electrical power. Set frequency omega to zero for DC modelling.

    This type is fully defined when 'I_rms', 'V_rms', 'X', and 'omega' attributes are set.
    """

    @property
    def _power(self):
        return self.V_rms * self.I_rms

    _I_rms = Quantity(np.nan, "A")
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
        return self._I_rms

    @I_rms.setter
    def I_rms(self, value):
        self._I_rms = Quantity(value, "A")

    @property
    def C(self) -> Quantity:
        """Equivalent electrical capacitance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore the divide by zero error from direct current
            return 1 / self.X / self.omega

    @C.setter
    def C(self, value):
        self.X = 1 / Quantity(value, "F") / self.omega

    @property
    def L(self) -> Quantity:
        """Equivalent electrical inductance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore the divide by zero error from direct current
            return self.X / self.omega

    @L.setter
    def L(self, value):
        self.X = Quantity(value, "H") * self.omega

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
        self.phi = np.arctan2(Quantity(value, "var"), self.power)

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
    """
    Mechanical power delivered through a rotating shaft.
    """

    @property
    def _power(self):
        return self.T * self.omega

    _T = Quantity(np.nan, "N m")
    _omega = Quantity(np.nan, "rad s^-1")

    @property
    def T(self) -> Quantity:
        """Rotational torque."""
        return self._T

    @T.setter
    def T(self, value):
        self._T = Quantity(value, "N m")

    @property
    def omega(self) -> Quantity:
        """Angular velocity."""
        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = Quantity(value, "rad s^-1")

    @property
    def nu(self):
        """Rotational frequency, i.e. the frequency with which one full rotation is completed."""
        return self.omega / (2 * np.pi)

    @nu.setter
    def nu(self, value):
        self.omega = value * (2 * np.pi)


class Thermal(AbstractPower):
    """
    Thermal power, i.e. power transferred through mass transport.
    """
    pass


class Radiant(AbstractPower):
    """
    Electromagnetical power, i.e. power transferred via electromagnetic irradiance.
    """
    pass


class Fluid(AbstractPower):
    """
    Fluidal power, i.e. the product of stagnation pressure and volumetric flow rate.

    This type is fully defined when 'Mach', 'mdot', and fluid 'state' and attributes are set. Optionally, Mach number
    may be omitted, in which case the fluid has no length or area scale.
    """

    @property
    def _power(self):
        # The total fluid power should come from the product of stagnation pressure and flow rate. Consider that in a
        #   U-tube of water static pressure contributes to the instantaneous pressure head, whereas the dynamic pressure
        #   is responsible for velocity head. Assuming no contribution from elevation head, their sum is the total head.
        total_power = np.nansum(
            self.power_pressure,
            self.power_velocity
        )
        return total_power

    @property
    def power_pressure(self):
        return self.state.pressure * self.Vdot

    @property
    def power_velocity(self):
        # Compute dynamic pressure
        q = 0.5 * self.state.density * self.u ** 2

        dynamic_power = q * self.Vdot
        return dynamic_power

    _Mach: float = 0
    _mdot = Quantity(0, "kg s^-1")
    _state: FluidState = None

    @property
    def Mach(self) -> float:
        """The Mach number of the flow."""
        return self._Mach

    @Mach.setter
    def Mach(self, value):
        self._Mach = float(value)

    @property
    def mdot(self) -> float:
        """Mass flow rate."""
        return self._mdot

    @mdot.setter
    def mdot(self, value):
        self._mdot = Quantity(value, "kg s^-1")

    @property
    def state(self) -> FluidState:
        """Returns an object describing the static fluid state."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def Vdot(self) -> Quantity:
        """Volumetric flow rate."""
        return self.mdot / self.state.density

    @Vdot.setter
    def Vdot(self, value):
        self.mdot = self.state.density * Quantity(value, "m^3 s^-1")

    @property
    def u(self):
        """Fluid velocity."""
        return self.Mach * self.state.speed_of_sound

    @u.setter
    def u(self, value):
        self.Mach = Quantity(value, "m s^-1") / self.state.speed_of_sound

    @property
    def q(self):
        """Dynamic pressure."""
        return 0.5 * self.state.density * self.u ** 2

    @q.setter
    def q(self, value):
        self.u = (2 * Quantity(value, "Pa") / self.state.density) ** 0.5

    @property
    def total_density(self):
        gamma = self.state.specific_heat_ratio
        T_Tt = self.state.temperature / self.total_temperature
        r_rt = T_Tt ** (1 / (gamma - 1))
        rt = self.state.density / r_rt
        return rt

    @total_density.setter
    def total_density(self, value):
        gamma = self.state.specific_heat_ratio
        rt = Quantity(value, "kg m^-3")
        r_rt = self.state.density / rt
        T_Tt = r_rt ** (gamma - 1)
        self.total_temperature = self.state.temperature / T_Tt

    @property
    def total_enthalpy(self):
        ht = self.state.specific_enthalpy + self.u ** 2 / 2
        return ht

    @total_enthalpy.setter
    def total_enthalpy(self, value):
        ht = Quantity(value, "J kg^-1")
        self.u = ((ht - self.state.specific_enthalpy) * 2) ** 0.5

    @property
    def total_pressure(self):
        # For adiabatic (no heat addition or rejection) and isentropic flows (no entropy gain)
        gamma = self.state.specific_heat_ratio
        T_Tt = self.state.temperature / self.total_temperature
        p_pt = T_Tt ** (gamma / (gamma - 1))
        pt = self.state.pressure / p_pt
        return pt

    @total_pressure.setter
    def total_pressure(self, value):
        gamma = self.state.specific_heat_ratio
        pt = Quantity(value, "Pa")
        p_pt = self.state.pressure / pt
        T_Tt = p_pt ** ((gamma - 1) / gamma)
        self.total_temperature = self.state.temperature / T_Tt

    @property
    def total_temperature(self):
        # For adiabatic flows (no heat addition or rejection)
        T_Tt = (1 + (self.state.specific_heat_ratio - 1) / 2 * self.Mach ** 2) ** -1
        Tt = self.state.temperature / T_Tt
        return Tt

    @total_temperature.setter
    def total_temperature(self, value):
        Tt = Quantity(value, "K")
        T_Tt = self.state.temperature / Tt
        self.Mach = (2 / (self.state.specific_heat_ratio - 1) * (1 / T_Tt - 1)) ** 0.5


class IOCollection:
    """A simple class for collecting together common IO types."""

    def __init__(self):
        self._chemical = []
        self._electrical = []
        self._mechanical = []
        self._thermal = []
        self._radiant = []
        self._fluid = []

    @property
    def chemical(self) -> list[Chemical]:
        return self._chemical

    @property
    def electrical(self) -> list[Electrical]:
        return self._electrical

    @property
    def mechanical(self) -> list[Mechanical]:
        return self._mechanical

    @property
    def thermal(self) -> list[Thermal]:
        return self._thermal

    @property
    def radiant(self) -> list[Radiant]:
        return self._radiant

    @property
    def fluid(self) -> list[Fluid]:
        return self._fluid


def collect(*powers: AbstractPower) -> IOCollection:
    """
    Given instances of classes which inherit AbstractPower, collect similar types of power.

    Args:
        *powers: Fully-defined instances of classes that inherit from AbstractPower.

    Returns:
        An IOCollection object that permits easy access to each of the powers passed as an argument to this function.

    """
    collection = IOCollection()
    allowed_types = [attr for attr in dir(collection) if not attr.startswith("_")]

    for power in powers:
        power_type = type(power).__name__.lower()

        if power_type not in allowed_types:
            error_msg = f"'{type(power).__name__}' is not allowed (not deemed to be any of the valid {allowed_types=})"
            raise TypeError(error_msg)

        setattr(collection, f"_{power_type}", getattr(collection, power_type) + [power])

    return collection
