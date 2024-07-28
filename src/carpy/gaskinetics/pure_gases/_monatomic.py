"""Module containing models of monatomic gases."""
from carpy.chemistry import Structure, ChemicalSpecies
from carpy.gaskinetics.pure_gases import PureGas

__all__ = ["argon", "helium", "krypton", "xenon"]
__author__ = "Yaseen Reza"

argon = PureGas(species=ChemicalSpecies(structures=Structure.from_condensed_formula("Ar")))
helium = PureGas(species=ChemicalSpecies(structures=Structure.from_condensed_formula("He")))
krypton = PureGas(species=ChemicalSpecies(structures=Structure.from_condensed_formula("Kr")))
xenon = PureGas(species=ChemicalSpecies(structures=Structure.from_condensed_formula("Xe")))
