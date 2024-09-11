from carpy.powerplant._io import IOType
from carpy.powerplant.modules import PlantModule

__all__ = ["CPreactor"]
__author__ = "Yaseen Reza"


class CPreactor(PlantModule):
    """
    Constant pressure through-flow reactor. Energy from chemical heat release is added to incident flow.
    """

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=(IOType.Fluid, IOType.Chemical),
            out_types=IOType.Fluid,
        )

        # Lower limit of zero as the fuel flow rate can drop to zero (no heat addition).
        self._admit_low = 0

    def forward(self, *inputs) -> tuple[IOType.AbstractPower, ...]:
        # Input checks
        inputs += tuple(self.inputs)
        assert len(inputs) == 2, f"{type(self).__name__} is expecting exactly two inputs (got {inputs})"
        assert [isinstance(input, self.inputs.legal_types) for input in inputs], f"{self.inputs.legal_types=}"
        assert type(inputs[0]) is not type(inputs[1]), f"expected inputs to be each of one of {self.inputs.legal_types}"
        inputs = IOType.collect(*inputs)

        # Unpack input
        fluid_in: IOType.Fluid = inputs.fluid[0]
        chem_in: IOType.Chemical = inputs.chemical[0]

        return
