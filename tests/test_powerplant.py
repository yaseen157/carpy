"""Tests for library powerplant analysis methods."""
import unittest

from carpy.physicalchem import species, fluids, eostate, UnreactiveFluidModel
from carpy.powerplant import IOType, modules
from carpy.utility import Quantity


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
            Mach=(speed := 90) / gas_state.speed_of_sound,  # unit circle, 90 m/s
            Vdot=3.141592 * speed
        )

        # The flow is passed into the diffuser
        my_diffuser = modules.Diffuser0d()
        my_diffuser.inputs.add(freestream_capture)
        diffuser_exit, = my_diffuser.forward()

        # Qualitative checks
        self.assertIsInstance(diffuser_exit, type(freestream_capture))
        self.assertLess(diffuser_exit.Mach, freestream_capture.Mach)
        self.assertLess(diffuser_exit.Vdot, freestream_capture.Vdot)
        self.assertGreater(diffuser_exit.state.pressure, freestream_capture.state.pressure)
        self.assertGreater(diffuser_exit.state.temperature, freestream_capture.state.temperature)
        return

    def test_axialcompressor0d(self):
        """Check the operation of a fluid compressor."""

        # Define a fluid
        gas_model = UnreactiveFluidModel()
        gas_model.X = {species.nitrogen(): 78, species.oxygen(): 21}

        # The fluid is given a state and assigned to a flow
        gas_state = gas_model(p=104290, T=291)
        diffuser_exit = IOType.Fluid(
            state=gas_state,
            Mach=0.146,
            Vdot=277
        )

        # Define mechanical work
        shaftpower = IOType.Mechanical()
        shaftpower.nu = 50  # Hertz
        shaftpower.T = 15000  # Newton-metres

        # Pass flow to a compression stage
        my_stage = modules.AxialCompressorStage0d()
        my_stage.inputs.add(shaftpower)
        stage_exit, = my_stage.forward(diffuser_exit)

        # Qualitative checks
        self.assertIsInstance(stage_exit, type(diffuser_exit))
        self.assertLess(stage_exit.Vdot, diffuser_exit.Vdot)
        self.assertEqual(stage_exit.mdot, diffuser_exit.mdot)
        self.assertGreater(stage_exit.state.pressure, diffuser_exit.state.pressure)
        self.assertGreater(stage_exit.state.temperature, diffuser_exit.state.temperature)
        return

    def test_combustor(self):
        # Define a fluid
        gas_model = UnreactiveFluidModel()
        gas_model.X = {species.nitrogen(): 78, species.oxygen(): 21}

        # The fluid is given a state and assigned to a flow
        gas_state = gas_model(p=116112, T=302)
        compressor_exit = IOType.Fluid(
            state=gas_state,
            Mach=0.18,
            Vdot=258
        )

        # Add fuel
        fuel_model = UnreactiveFluidModel()
        fuel_model.X = {species.methane(): 1}
        # fuel_model = fluids.Jet_A(eos_class=eostate.SRKmP)
        fuel_state = fuel_model(p=101325, T=300)
        injection_flow = IOType.Fluid(
            state=fuel_state,
            mdot=Quantity(3e3, "lb hr^-1")
        )

        # Set up the combustor and pass the flow to it
        mycombustor = modules.ConstPCombustor()
        mycombustor.injector.inputs.add(injection_flow)
        combustor_exit, = mycombustor.forward(compressor_exit)

        return

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
