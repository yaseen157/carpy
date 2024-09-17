from carpy.physicalchem import UnreactiveFluidModel, VanderWaals, species
from carpy.powerplant._io import IOType
from carpy.powerplant.modules import PlantModule

__all__ = ["CPreactor"]
__author__ = "Yaseen Reza"


class FuelModels:

    @staticmethod
    def Jet_A() -> UnreactiveFluidModel:
        """
        A representative model for Jet-A aviation turbine fuel.

        This model is based on the findings of Huber et al. (2010) in creating a surrogate model for Jet-A-4658, which
        is a composite mixture of several Jet-A samples from different manufacturers.

        Notes:
            Due to the infancy of the library, we do not yet have the ability to compose jet fuel from its constituent
            species. As a result, this is at best a representation of the average properties of the fluid.

        References:
            Huber, M.L., Lemmon, E.W. and Bruno, T.J., 2010. Surrogate mixture models for the thermophysical properties
            of aviation fuel Jet-A. Energy & Fuels, 24(6), pp.3565-3571. https://doi.org/10.1021/ef100208c

        """
        fluid_model = UnreactiveFluidModel(eos_class=VanderWaals)
        fluid_model.X = {
            species.n_heptylcyclohexane(): 0.279,
            species._1_methyldecalin(): 0.013,
            species._5_methylnonane(): 0.165,
            species._2_methyldecane(): 0.154,
            species.n_tetradecane(): 0.057,
            species.n_hexadecane(): 0.033,
            species.ortho_xylene(): 0.071,
            species.tetralin(): 0.228
        }
        return fluid_model


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


if __name__ == "__main__":
    # fuel = FuelModels.Jet_A()
    # print(fuel.density(p=83e3, T=50))

    water = UnreactiveFluidModel(eos_class=VanderWaals)
    water.X = {species.water(): 1}
    print(water.density(p=101325, T=373))
