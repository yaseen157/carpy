import periodictable as pt

from carpy.physicalchem import ChemicalSpecies, UnreactiveFluidModel, species
from carpy.powerplant import IOType
from carpy.powerplant.modules import PlantModule, FTypeGovernor
from carpy.utility import Quantity

__all__ = ["ConstPCombustor"]
__author__ = "Yaseen Reza"


def find_CHO_species(composition_dict: dict[ChemicalSpecies, float]) -> list[ChemicalSpecies]:
    """Identify chemical species in a chemical composition that are only made of carbon, hydrogen, and oxygen."""
    is_CHO = []
    for i, species in enumerate(composition_dict):
        if set(species.composition_formulaic) - {pt.C, pt.H, pt.O}:  # noqa
            pass
        else:
            is_CHO.append(species)
    return is_CHO


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

    def forward(self, *inputs) -> tuple[IOType.AbstractFlow, ...]:
        # Input checks
        inputs += tuple(self.inputs)
        assert len(inputs) == 1, f"{type(self).__name__} is expecting exactly one input (got {inputs})"
        assert [isinstance(_input, self.inputs.legal_types) for _input in inputs], f"{self.inputs.legal_types=}"

        # Unpack input
        primary_flow, = inputs
        secondary_flow, = self.injector.forward()

        # Determine the carbon, hydrogen, and oxygen content of each flow, scaled by the volumetric (molar) flow rate
        CHO_ndot = dict()
        not_CHO_ndot = dict()

        # For each of the two flows...
        for flow in (primary_flow, secondary_flow):
            # ... identify the species that do and don't participate in combustion
            CHO_species = find_CHO_species(composition_dict=flow.state.X)

            # For the species that do participate in combustion
            for species_i in flow.state.X:

                species_mdot = flow.mdot * flow.state.Y[species_i]  # The mass flow rate of this species, given Y
                species_ndot = species_mdot / species_i.molar_mass  # The molar flow rate of this species

                # ndot of elements is the product of the species molar flow rate and number of atoms of that element
                if species_i in CHO_species:
                    for (element, n_atoms) in species_i.composition_formulaic.items():
                        CHO_ndot[element] = CHO_ndot.get(element, 0) + n_atoms * species_ndot
                else:
                    not_CHO_ndot[species_i] = species_ndot

        # Compute the number of output moles of CO2 and H2O
        # H_m C_n O_o --> X. H2O + Y. CO2 + Z. O2
        m = CHO_ndot.get(pt.H, Quantity(0, "mol s^-1"))  # noqa
        n = CHO_ndot.get(pt.C, Quantity(0, "mol s^-1"))  # noqa
        o = CHO_ndot.get(pt.O, Quantity(0, "mol s^-1"))  # noqa
        X = m / 2
        Y = n
        Z = (o - X - 2 * Y) / 2

        # Check if complete combustion is possible
        if Z < 0:
            error_msg = f"{type(self).__name__} has an excess of fuel, complete combustion is impossible"
            raise ValueError(error_msg)

        # Spawn output fluid. In a reaction, the number of moles are not conserved but the total mass is!
        H2O_mdot = X * species.water().molar_mass
        CO2_mdot = Y * species.carbon_dioxide().molar_mass
        O2_mdot = Z * species.oxygen().molar_mass
        not_CHO_mdot = {species_i: species_i.molar_mass * ndot for species_i, ndot in not_CHO_ndot.items()}

        product_model = UnreactiveFluidModel(eos_class=primary_flow.state.EOS.__class__)
        product_model.Y = {
            species.water(): H2O_mdot,
            species.carbon_dioxide(): CO2_mdot,
            species.oxygen(): O2_mdot,
            **not_CHO_mdot
        }
        # Provide an initial temperature, literally for the sake of argument
        product_state = product_model(p=primary_flow.state.pressure, T=primary_flow.state.temperature)
        product_flow = IOType.Fluid(
            state=product_state,
            mdot=(primary_flow.mdot + secondary_flow.mdot),
            Mach=primary_flow.Mach
        )

        power_released = sum((
            primary_flow.state.enthalpy_atomisation * primary_flow.ndot,  # Heat release in atomisation
            secondary_flow.state.enthalpy_atomisation * secondary_flow.ndot,  # Heat release in atomisation
            -product_flow.state.enthalpy_atomisation * product_flow.ndot  # Heat absorption in molecularisation(?)
        ))

        # Pack output
        self.outputs.clear()

        # TODO: Need to work out internal energy capacity to temperature conversion

        return NotImplemented
