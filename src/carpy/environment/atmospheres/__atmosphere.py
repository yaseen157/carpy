"""Module implementing the basis class for modelling static atmospheres."""
import typing

from carpy.gaskinetics import NonReactiveGasModel
from carpy.utility import Quantity

__all__ = ["StaticAtmosphereModel"]
__author__ = "Yaseen Reza"


class StaticAtmosphereModel:
    """
    Base class for all reference/standardised models of atmospheric state variables. Constituent methods are always
    functions of geometric altitude.

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

    @classmethod
    def temperature(cls, z) -> Quantity:
        """
        Args:
            z: Geometric altitude, in metres.

        Returns:
            Ambient temperature.

        """
        z = Quantity(z, "m")
        return cls._temperature(z=z)

    @classmethod
    def pressure(cls, z) -> Quantity:
        """
        Args:
            z: Geometric altitude, in metres.

        Returns:
            Ambient pressure.

        """
        z = Quantity(z, "m")
        return cls._pressure(z=z)

    def molar_volume(self, z) -> Quantity:
        """
        Computes the molar volume at the given altitude, i.e. the volume of space occupied by one mole of gas.

        Args:
            z: Geometric altitude, in metres.

        Returns:
            Ambient molar volume.

        """
        p = self.pressure(z=z)
        T = self.temperature(z=z)
        Vm = self._gas_model.molar_volume(p=p, T=T)
        return Vm

    def density(self, z: Quantity) -> Quantity:
        """
        Args:
            z: Geometric altitude, in metres.

        Returns:
            Ambient density.

        """
        p = self.pressure(z=z)
        T = self.temperature(z=z)
        rho = self._gas_model.density(p=p, T=T)
        return rho

    def _speed_of_sound(self, z: Quantity):
        p = self.pressure(z=z)
        T = self.temperature(z=z)
        a = self._gas_model.speed_of_sound(p=p, T=T)
        return a

    def speed_of_sound(self, z) -> Quantity:
        """
        Args:
            z: Geometric altitude, in metres.

        Returns:
            Ambient speed of sound.

        """
        return self._speed_of_sound(z=z)

    def dynamic_viscosity(self, z) -> Quantity:
        """
        Args:
            z: Geometric altitude, in metres.

        Returns:
            Ambient dynamic viscosity.

        """
        z = Quantity(z, "m")
        return self._dynamic_viscosity(z=z)

    def _kinematic_viscosity(self, z: Quantity):
        mu = self.dynamic_viscosity(z=z)
        rho = self.density(z=z)
        nu = mu / rho
        return nu

    def kinematic_viscosity(self, z) -> Quantity:
        """
        Args:
            z: Geometric altitude, in metres.

        Returns:
            Ambient kinematic viscosity.

        """
        z = Quantity(z, "m")
        return self._kinematic_viscosity(z=z)
