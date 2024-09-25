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

        # Determine the mass flow rate of reacting and non-reacting species
        mdot_reactants, mdot_cheminert = dict(), dict()
        for flow in (primary_flow, secondary_flow):
            for species_i, Yi in flow.state.Y.items():
                species_mdot = flow.mdot * Yi

                # If the reacting species contain any elements that are not of the set {C, H, O}, do not combust it
                if set(species_i.composition_formulaic) - {pt.C, pt.H, pt.O}:  # noqa
                    mdot_cheminert[species_i] = species_mdot
                else:
                    mdot_reactants[species_i] = species_mdot

        # Conserving mass flow rate of reactants, determine the mass (and molar) flow rate of mono-atoms (elements)
        mdot_elements = dict()
        for species_i, species_mdot in mdot_reactants.items():

            for element, count in species_i.composition_formulaic.items():
                mfrac = float(count * element.mass / species_i.molecular_mass.to("Da"))
                mdot_elements[element] = mdot_elements.get(element, 0) + species_mdot * mfrac

        # Convert mass flow of reactant elements into molar flow, and then determine the quantity of product produced
        # H_m C_n O_o --> X. H2O + Y. CO2 + Z. O2
        nu_H = mdot_elements.get(pt.H, Quantity(0, "mol s^-1")) / Quantity(pt.H.mass, "g mol^-1")  # noqa
        nu_C = mdot_elements.get(pt.C, Quantity(0, "mol s^-1")) / Quantity(pt.C.mass, "g mol^-1")  # noqa
        nu_O = mdot_elements.get(pt.O, Quantity(0, "mol s^-1")) / Quantity(pt.O.mass, "g mol^-1")  # noqa
        # Compute the stoichiometric coefficients of the products
        nu_H2O = nu_H / 2
        nu_CO2 = nu_C
        nu_O2 = (nu_O - nu_H2O - 2 * nu_CO2) / 2

        mdot_products = {
            species.water(): nu_H2O * species.water().molar_mass,
            species.carbon_dioxide(): nu_CO2 * species.carbon_dioxide().molar_mass,
            species.oxygen(): nu_O2 * species.oxygen().molar_mass
        }

        power_raised = sum([
                               -nu_H2O * species.water().enthalpy_atomisation,
                               -nu_CO2 * species.carbon_dioxide().enthalpy_atomisation,
                               -nu_O2 * species.oxygen().enthalpy_atomisation,
                           ] + [
                               (mdot / species_i.molar_mass) * species_i.enthalpy_atomisation
                               for species_i, mdot in mdot_reactants.items()
                           ])

        # TODO: Need to work out internal energy capacity to temperature conversion

        return NotImplemented
