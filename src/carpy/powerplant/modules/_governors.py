from carpy.powerplant import IOType
from carpy.powerplant.modules import PlantModule

__all__ = ["ETypeGovernor", "ElectronicSpeedControl", "ESC",
           "FTypeGovernor", "FuelInjector"]
__author__ = "Yaseen Reza"


class ETypeGovernor(PlantModule):
    """
    Electronic Speed Control (ESC) unit.

    Notes:
        With default instantiation, the power admitted is limited (by a throttle).

    """

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=IOType.Electrical,
            out_types=IOType.Electrical
        )
        raise NotImplementedError


ElectronicSpeedControl = ETypeGovernor
ESC = ETypeGovernor


class FTypeGovernor(PlantModule):
    """
    Fluid flow governance unit.
    """

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=IOType.Fluid,
            out_types=IOType.Fluid
        )

    def forward(self, *inputs) -> tuple[IOType.AbstractFlow, ...]:
        # Input checks
        inputs += tuple(self.inputs)
        assert len(inputs) == 1, f"{type(self).__name__} is expecting exactly one input (got {inputs})"
        assert [isinstance(input, self.inputs.legal_types) for input in inputs], f"{self.inputs.legal_types=}"

        # Unpack input
        fluid_in, = inputs

        # Do something? maybe? pressure drop?
        fluid_out = fluid_in

        # Pack output
        self.outputs.clear()
        self.outputs.add(fluid_out)
        return (fluid_out,)


FuelInjector = FTypeGovernor

# TODO: Investigate time varying fuel injection??: https://www.intechopen.com/chapters/19534
