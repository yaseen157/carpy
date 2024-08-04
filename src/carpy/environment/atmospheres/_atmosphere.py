"""Module implementing the basis class for modelling static atmospheres."""
import typing

from carpy.gaskinetics import NonReactiveGasModel
from carpy.utility import Quantity

__all__ = ["StaticAtmosphereModel"]
__author__ = "Yaseen Reza"


class StaticAtmosphereModel:
    """
    Base class for all reference/standardised models of atmospheric state variables. Constituent methods are always
    functions of geopotential altitude.

    The World Meteorological Organisation defines a standard atmosphere as "a hypothetical vertical distribution of
    atmospheric temperature, pressure and density."

    Notes:
        Base methods for molar volume, density, and speed of sound depend on the definition of a 'non-reactive gas
        model'. As such, each model derived from this class must either specify a composition for the private gas model
        attribute, or redefine the aforementioned base methods without reference to the model.

        The model's attribution is kept private as not all models will make use of the model, and in some cases, the
        model may not accurately reflect the composition of gases in the atmosphere if they are deemed to change with,
        for example, altitude.

    """
    _gas_model: NonReactiveGasModel
    _planet: str = None

    # Atmospheric profile functions of geometric altitude
    _temperature: typing.Callable
    _pressure: typing.Callable
    _dynamic_viscosity: typing.Callable

    def __init__(self, *args, **kwargs):
        self._gas_model = NonReactiveGasModel()

    def __getattr__(self, item):
        # print(f"hooked a call to {self.__repr__()}.{item}!!!")
        return super().__getattribute__(item)

    def __call__(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    def temperature(self, h) -> Quantity:
        """
        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Ambient temperature.

        """
        h = Quantity(h, "m")
        return self._temperature(h=h)

    def pressure(self, h) -> Quantity:
        """
        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Ambient pressure.

        """
        h = Quantity(h, "m")
        return self._pressure(h=h)

    def molar_volume(self, h) -> Quantity:
        """
        Computes the molar volume at the given altitude, i.e. the volume of space occupied by one mole of gas.

        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Ambient molar volume.

        """
        p = self.pressure(h=h)
        T = self.temperature(h=h)
        Vm = self._gas_model.molar_volume(p=p, T=T)
        return Vm

    def _density(self, h: Quantity) -> Quantity:
        Vm = self.molar_volume(h=h)
        rho = self._gas_model.molar_mass / Vm
        return rho

    def density(self, h) -> Quantity:
        """
        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Ambient density.

        """
        h = Quantity(h, "m")
        return self._density(h=h)

    def _speed_of_sound(self, h: Quantity):
        p = self.pressure(h=h)
        T = self.temperature(h=h)
        a = self._gas_model.speed_of_sound(p=p, T=T)
        return a

    def speed_of_sound(self, h) -> Quantity:
        """
        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Ambient speed of sound.

        """
        return self._speed_of_sound(h=h)

    def dynamic_viscosity(self, h) -> Quantity:
        """
        Args:
            h: Geopotential altitude, in metres.

        Returns:
            Ambient dynamic viscosity.

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
            Ambient kinematic viscosity.

        """
        h = Quantity(h, "m")
        return self._kinematic_viscosity(h=h)
