"""Tests for library physicalchem methods."""
import unittest

from carpy.physicalchem._atom import Atom
from carpy.physicalchem._chemical_structure import Structure
from carpy.utility import constants as co


class AtomicProperties(unittest.TestCase):
    """Check that we can get properties of atoms."""

    def test_hydrogen(self):
        atom = Atom("H")
        self.assertEqual(atom.atomic_number, 1)
        self.assertEqual(atom.atomic_charge, 0)
        self.assertAlmostEqual(atom.atomic_mass, 1.00e-3 / co.PHYSICAL.N_A.x)
        self.assertEqual(atom.electrons.pt_group, 1)
        return

    def test_boron(self):
        atom = Atom("B")
        self.assertEqual(atom.atomic_number, 5)
        self.assertEqual(atom.atomic_charge, 0)
        self.assertAlmostEqual(atom.atomic_mass, 10.8e-3 / co.PHYSICAL.N_A.x)
        self.assertEqual(atom.electrons.pt_group, 13)
        return

    def test_carbon(self):
        atom = Atom("C")
        self.assertEqual(atom.atomic_number, 6)
        self.assertEqual(atom.atomic_charge, 0)
        self.assertAlmostEqual(atom.atomic_mass, 12.0e-3 / co.PHYSICAL.N_A.x)
        self.assertEqual(atom.electrons.pt_group, 14)
        return

    def test_magnesium(self):
        atom = Atom("Mg")
        self.assertEqual(atom.atomic_number, 12)
        self.assertEqual(atom.atomic_charge, 0)
        self.assertAlmostEqual(atom.atomic_mass, 24.3e-3 / co.PHYSICAL.N_A.x)
        self.assertEqual(atom.electrons.pt_group, 2)
        return

    def test_chlorine(self):
        atom = Atom("Cl")
        self.assertEqual(atom.atomic_number, 17)
        self.assertEqual(atom.atomic_charge, 0)
        self.assertAlmostEqual(atom.atomic_mass, 35.5e-3 / co.PHYSICAL.N_A.x)
        self.assertEqual(atom.electrons.pt_group, 17)
        return

    def test_chromium(self):
        atom = Atom("Cr")
        self.assertEqual(atom.atomic_number, 24)
        self.assertEqual(atom.atomic_charge, 0)
        self.assertAlmostEqual(atom.atomic_mass, 52e-3 / co.PHYSICAL.N_A.x)
        self.assertEqual(atom.electrons.pt_group, 6)
        return

    def test_xenon(self):
        atom = Atom("Xe")
        self.assertEqual(atom.atomic_number, 54)
        self.assertEqual(atom.atomic_charge, 0)
        self.assertAlmostEqual(atom.atomic_mass, 131e-3 / co.PHYSICAL.N_A.x)
        self.assertEqual(atom.electrons.pt_group, 18)
        return


class MolecularStructures(unittest.TestCase):
    """Test the molecular structures."""

    def test_monatomic(self):
        argon = Structure.from_condensed_formula("Ar")
        return

    def test_diatomic(self):
        # Spawn nitrogen
        nitrogen = Structure.from_condensed_formula("N2")

        # Nitrogen has one bond
        self.assertEqual(len(nitrogen.bonds), 1)
        # ...with order three
        bond = nitrogen.bonds.pop()
        self.assertEqual(bond.order, 3)
        # Two atoms partake in it
        self.assertEqual(len(bond.atoms), 2)
        # ...and are the constituents of the molecule
        self.assertEqual(set(bond.atoms), nitrogen.atoms)
        for atom in nitrogen.atoms:
            # Homonuclear diatomic has an oxidation state of zero
            self.assertEqual(atom.oxidation_state, 0)
            # ...and similarly no charge
            self.assertEqual(atom.atomic_charge, 0)

        # Nitrogen has the following characteristic temperatures
        self.assertAlmostEqual(nitrogen.theta_rot[0], 2.86, places=2)  # noqa
        self.assertAlmostEqual(nitrogen.theta_vib[bond], 3340, delta=100)  # noqa
        self.assertAlmostEqual(nitrogen.theta_diss, 113000, delta=1000)  # noqa

        # Spawn carbon monoxide
        carbonmonoxide = Structure.from_condensed_formula("CO")

        # Carbon monoxide has one bond
        self.assertEqual(len(carbonmonoxide.bonds), 1)
        # ...with order three
        bond = carbonmonoxide.bonds.pop()
        self.assertEqual(bond.order, 3)
        # Two atoms partake in it
        self.assertEqual(len(bond.atoms), 2)
        # ...and are the constituents of the molecule
        self.assertEqual(set(bond.atoms), carbonmonoxide.atoms)
        for atom in carbonmonoxide.atoms:
            # Carbon in the molecule has an oxidation state of two
            if atom.symbol == "C":
                self.assertEqual(atom.oxidation_state, 2)
            # ...which otherwise implies an oxidation state of negative two
            else:
                self.assertEqual(atom.oxidation_state, -2)

            # The dative covalent bond means the carbon effectively picks up an extra electron
            if atom.symbol == "C":
                self.assertEqual(atom.atomic_charge, -1)
            # ...meaning the oxygen is effectively losing one and "oxidising"
            else:
                self.assertEqual(atom.atomic_charge, 1)

        return


if __name__ == "__main__":
    unittest.main()