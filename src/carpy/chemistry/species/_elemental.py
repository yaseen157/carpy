"""Module containing chemical species definitions for elemental (at standard temperature and pressure) gases."""
from carpy.chemistry import Structure, ChemicalSpecies

__all__ = ["argon", "chlorine", "fluorine", "helium", "hydrogen", "krypton", "neon", "nitrogen", "oxygen", "xenon"]
__author__ = "Yaseen Reza"

argon = ChemicalSpecies(structures=Structure.from_condensed_formula("Ar"))
chlorine = ChemicalSpecies(structures=Structure.from_condensed_formula("Cl2"))
fluorine = ChemicalSpecies(structures=Structure.from_condensed_formula("F2"))
helium = ChemicalSpecies(structures=Structure.from_condensed_formula("He"))
hydrogen = ChemicalSpecies(structures=Structure.from_condensed_formula("H2"))
krypton = ChemicalSpecies(structures=Structure.from_condensed_formula("Kr"))
neon = ChemicalSpecies(structures=Structure.from_condensed_formula("Ne"))
nitrogen = ChemicalSpecies(structures=Structure.from_condensed_formula("N2"))
oxygen = ChemicalSpecies(structures=Structure.from_condensed_formula("O2"))
radon = ChemicalSpecies(structures=Structure.from_condensed_formula("O2"))
xenon = ChemicalSpecies(structures=Structure.from_condensed_formula("Xe"))
