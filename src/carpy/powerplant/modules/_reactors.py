from carpy.powerplant._io import IOType
from carpy.powerplant.modules import PlantModule
from carpy.powerplant.modules._governors import FTypeGovernor

__all__ = ["ConstPCombustor"]
__author__ = "Yaseen Reza"


class ConstPCombustor(PlantModule):
    """
    Constant pressure through-flow (continuous) reactor. Energy from chemical heat release is added to incident flow.
    """
    _injector: FTypeGovernor

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=IOType.Fluid,
            out_types=IOType.Fluid,
        )

        self._injector = FTypeGovernor(name=name)

    @property
    def injector(self) -> FTypeGovernor:
        return self._injector

    def forward(self, *inputs) -> tuple[IOType.AbstractPower, ...]:
        # Input checks
        inputs += tuple(self.inputs)
        assert len(inputs) == 1, f"{type(self).__name__} is expecting exactly one input (got {inputs})"
        assert [isinstance(input, self.inputs.legal_types) for input in inputs], f"{self.inputs.legal_types=}"

        # Unpack input
        oxidiser_in, = inputs
        fuel_in, = self.injector.forward()

        # Determine the molar content of the input fuel

        # Determine the molar oxygen content of the input oxidiser

        # Check if complete combustion is possible

        # Spawn output fluid

        # Pack output
        self.outputs.clear()

        return NotImplemented
