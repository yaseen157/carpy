"""Tests for library powerplant analysis methods."""
import unittest

from carpy.powerplant import IOType, modules, PowerNetwork


class Networking(unittest.TestCase):

    def test_simple_solar(self):
        """A linear, acyclic solar network with no time dependencies."""
        # Define components of network
        my_batt = modules.Battery()
        my_cell = modules.PVCell()
        my_motor = modules.ElectricMotor()

        # Define network connections
        my_batt <<= my_cell
        my_batt >>= my_motor

        # Set network performance targets
        my_motor >>= IOType.Mechanical(power=350)

        pn = PowerNetwork(my_batt)
        return


if __name__ == "__main__":
    unittest.main()
