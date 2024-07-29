"""Module implementing the basis class for atmospheric modelling."""
from carpy.chemistry import EquationOfState, IdealGas
from carpy.utility import Quantity

__all__ = ["StaticAtmosphereModel"]
__author__ = "Yaseen Reza"


# TODO: Decide whether it makes sense to have "class Climate: def __init__(self, atm: StaticAtmosphereModel, ...)"

class StaticAtmosphereModel:
    """Base class for all reference/standardised models of atmospheric state variables."""

    def __init__(self, equation_of_state: EquationOfState = None):
        self._equation_of_state = IdealGas() if equation_of_state is None else equation_of_state
        return

    def _temperature(self, h: Quantity):
        error_msg = f"Sorry, the {type(self).__name__} atmosphere model has not yet implemented this parameter."
        raise NotImplementedError(error_msg)

    def temperature(self, h) -> Quantity:
        """
        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Local atmospheric temperature.

        """
        h = Quantity(h, "m")
        return self._temperature(h=h)

    def _pressure(self, h: Quantity):
        error_msg = f"Sorry, the {type(self).__name__} atmosphere model has not yet implemented this parameter."
        raise NotImplementedError(error_msg)

    def pressure(self, h) -> Quantity:
        """
        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Local atmospheric pressure.

        """
        h = Quantity(h, "m")
        return self._pressure(h=h)

    def molar_volume(self, h) -> Quantity:
        """
        Computes the molar volume at the given altitude, i.e. the volume of space occupied by one mole of gas.

        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Local atmospheric molar volume.

        """
        p = self.pressure(h=h)
        T = self.temperature(h=h)
        Vm = self._equation_of_state.molar_volume(p=p, T=T)
        return Vm

    def _density(self, h: Quantity):
        error_msg = f"Sorry, the {type(self).__name__} atmosphere model has not yet implemented this parameter."
        raise NotImplementedError(error_msg)

    def density(self, h) -> Quantity:
        """
        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Local atmospheric density.

        """
        h = Quantity(h, "m")
        return self._density(h=h)

    def _speed_of_sound(self, h: Quantity):
        error_msg = f"Sorry, the {type(self).__name__} atmosphere model has not yet implemented this parameter."
        raise NotImplementedError(error_msg)

    def speed_of_sound(self, h) -> Quantity:
        """
        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Local atmospheric speed of sound.

        """
        return self._speed_of_sound(h=h)

    def _dynamic_viscosity(self, h: Quantity):
        error_msg = f"Sorry, the {type(self).__name__} atmosphere model has not yet implemented this parameter."
        raise NotImplementedError(error_msg)

    def dynamic_viscosity(self, h) -> Quantity:
        """
        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Local atmospheric dynamic viscosity.

        """
        h = Quantity(h, "m")
        return self._dynamic_viscosity(h=h)

    def _kinematic_viscosity(self, h: Quantity):
        mu = self.dynamic_viscosity(h=h)
        rho = self.density(h=h)
        nu = mu / rho
        return nu

    def kinematic_viscosity(self, h) -> Quantity:
        """
        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Local atmospheric kinematic viscosity.

        """
        h = Quantity(h, "m")
        return self._kinematic_viscosity(h=h)
