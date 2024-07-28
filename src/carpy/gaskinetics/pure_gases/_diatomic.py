"""Module containing models of diatomic gases."""
from carpy.chemistry import Structure, ChemicalSpecies
from carpy.gaskinetics.pure_gases import PureGas

__all__ = ["chlorine", "fluorine", "hydrogen", "nitrogen", "oxygen"]
__author__ = "Yaseen Reza"

chlorine = PureGas(species=ChemicalSpecies(structures=Structure.from_condensed_formula("Cl2")))
fluorine = PureGas(species=ChemicalSpecies(structures=Structure.from_condensed_formula("F2")))
hydrogen = PureGas(species=ChemicalSpecies(structures=Structure.from_condensed_formula("H2")))
nitrogen = PureGas(species=ChemicalSpecies(structures=Structure.from_condensed_formula("N2")))
oxygen = PureGas(species=ChemicalSpecies(structures=Structure.from_condensed_formula("O2")))
