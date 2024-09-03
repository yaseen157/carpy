"""Tests for library powerplant analysis methods."""
import unittest

from carpy.physicalchem import species, UnreactiveFluidModel
from carpy.powerplant import IOType, modules, PowerNetwork


class PlantModules(unittest.TestCase):

    def test_diffuser0d(self):
        """Check the operation of an adiabatic diffuser."""

        # Define a fluid
        gas_model = UnreactiveFluidModel()
        gas_model.X = {species.nitrogen(): 78, species.oxygen(): 21}

        # The fluid is given a state and assigned to a flow
        gas_state = gas_model(p=101325, T=288.15)
        freestream_capture = IOType.Fluid(
            state=gas_state,
            Mach=(speed := 220) / gas_state.speed_of_sound,  # unit circle, 220 m/s
            Vdot=3.141592 * speed
        )

        # The flow is passed into the diffuser
        my_diffuser = modules.Diffuser0d()
        diffuser_exit, = my_diffuser.forward(freestream_capture)

        self.assertIsInstance(diffuser_exit, type(freestream_capture))
        self.assertLess(diffuser_exit.Mach, freestream_capture.Mach)
        self.assertLess(diffuser_exit.Vdot, freestream_capture.Vdot)
        self.assertGreater(diffuser_exit.state.pressure, freestream_capture.state.pressure)
        self.assertGreater(diffuser_exit.state.temperature, freestream_capture.state.temperature)
        return

    def test_axialpump(self):
        """Check the operation of a fluid compressor."""

        # Define a fluid
        gas_model = UnreactiveFluidModel()
        gas_model.X = {species.nitrogen(): 78, species.oxygen(): 21}

        # The fluid is given a state and assigned to a flow
        gas_state = gas_model(p=119046.82753515, T=310.90963859)
        diffuser_exit = IOType.Fluid(
            state=gas_state,
            Mach=0.13894453308458346,
            Vdot=634.72676389
        )

        # Define mechanical work
        shaftpower = IOType.Mechanical()
        shaftpower.nu = 50  # 50 Hertz
        shaftpower.T = 150  # 150 Newton-metres

        # The flow is passed into the compressor
        my_compressor = modules.AxialPump()
        compressor_exit, = my_compressor.forward(diffuser_exit, shaftpower)

#
#     def test_combustor(self):
#         """A simple combustor heat addition test."""
#         combustor = modules.Combustor()
#
#         fluid_in = IOType.Fluid()
#         fluid_in.pressure = 6e5
#         fluid_in.mdot = 2.4
#         fluid_in.Vdot = 3
#
#         fuel = IOType.Chemical()
#         fuel.mdot = 0.1
#         fuel.CV = 43e6
#
#         combustor <<= fluid_in
#         combustor <<= fuel
#         combustor.forward()
#         return
#
#     def test_simple_solar(self):
#         """A linear, acyclic solar network with no time dependencies."""
#         # Define components of network
#         pv_cell1, pv_cell2 = modules.PVCell("SolarCell1"), modules.PVCell("SolarCell2")
#         my_batt = modules.Battery("Batt")
#         my_esc = modules.ElectronicSpeedControl("ESC")
#         my_motor = modules.ElectricMotor("Motor")
#
#         # Define network connections
#         my_batt <<= pv_cell1
#         my_batt <<= pv_cell2
#         my_esc <<= my_batt
#         my_motor <<= my_esc
#
#         # Set network performance targets
#         pv_cell1 <<= IOType.Radiant(power=200)
#         pv_cell2 <<= IOType.Radiant(power=200)
#         my_motor >>= IOType.Mechanical(power=350)
#
#         pn = PowerNetwork(my_batt)
#         pn.solve()
#         return
#
#
# if __name__ == "__main__":
#     unittest.main()
