"""Module pre-defining important fluids and gases for quick user use."""
import cantera as ct

from carpy.fluiddynamics._fluidstates import Fluid

__all__ = ["Fluids"]


class Fluids(object):
    """A collection of FLuid objects."""

    class PerfectGas:
        """Perfect (a.k.a. calorically perfect) gases."""

        @staticmethod
        def CarbonDioxide() -> Fluid:
            """Carbon dioxide at standard temperature and pressure."""
            fluid = Fluid.from_gasmodel_perfect("CO2")
            return fluid

        CO2 = CarbonDioxide

        @staticmethod
        def Hydrogen() -> Fluid:
            """Hydrogen at standard temperature and pressure."""
            fluid = Fluid.from_gasmodel_perfect("H2")
            return fluid

        H2 = Hydrogen

    class PureFluid:
        """Pure, homogeneous fluids."""

        @staticmethod
        def CarbonDioxide() -> Fluid:
            """Carbon dioxide at standard temperature and pressure."""
            model = ct.CarbonDioxide()
            return Fluid.from_cantera_fluid(model)

        CO2 = CarbonDioxide

        @staticmethod
        def Heptane() -> Fluid:
            """Heptane at standard temperature and pressure."""
            model = ct.Heptane()
            return Fluid.from_cantera_fluid(model)

        C7H16 = Heptane

        @staticmethod
        def HFC134a() -> Fluid:
            """HFC134a refrigerant at standard temperature and pressure."""
            model = ct.Hfc134a()
            return Fluid.from_cantera_fluid(model)

        R134a = HFC134a

        @staticmethod
        def Hydrogen() -> Fluid:
            """Hydrogen at standard temperature and pressure."""
            model = ct.Hydrogen()
            return Fluid.from_cantera_fluid(model)

        H2 = Hydrogen

        @staticmethod
        def Methane() -> Fluid:
            """Methane at standard temperature and pressure."""
            model = ct.Methane()
            return Fluid.from_cantera_fluid(model)

        CH4 = Methane

        @staticmethod
        def Nitrogen() -> Fluid:
            """Nitrogen at standard temperature and pressure."""
            model = ct.Nitrogen()
            return Fluid.from_cantera_fluid(model)

        N2 = Nitrogen

        @staticmethod
        def Oxygen() -> Fluid:
            """Oxygen at standard temperature and pressure."""
            model = ct.Oxygen()
            return Fluid.from_cantera_fluid(model)

        O2 = Oxygen

        @staticmethod
        def Water() -> Fluid:
            """Water at standard temperature and pressure."""
            model = ct.Water()
            return Fluid.from_cantera_fluid(model)

        H2O = Water

    class GRI30:
        """Gases based on GRI-Mech 3.0 combustion model."""

        @staticmethod
        def Air() -> Fluid:
            """Air with standard composition, temperature, and pressure."""
            mech = "gri30.yaml"
            compositionX = {
                "N2": 78.084,
                "O2": 20.946,
                "Ar": 0.9340,
                "CO2": 0.0407,
                "CH4": 0.00018,
                "H2": 0.000055
            }
            return Fluid.from_cantera_mech(mech, compositionX)

    class GRI30highT:
        """
        Gases based on GRI-Mech 3.0 combustion model, with high temperature
        modifications.
        """

        @staticmethod
        def Air() -> Fluid:
            """Air with standard composition, temperature, and pressure."""
            mech = "gri30_highT.yaml"
            compositionX = {
                "N2": 78.084,
                "O2": 20.946,
                "Ar": 0.9340,
                "CO2": 0.0407,
                "CH4": 0.00018,
                "H2": 0.000055
            }
            return Fluid.from_cantera_mech(mech, compositionX)

    class Air:
        """Default air model, as given in Cantera's air.yaml."""

        @staticmethod
        def Air() -> Fluid:
            """Air with standard composition, temperature, and pressure."""
            mech = "air.yaml"
            fluid = Fluid.from_cantera_mech(mech)
            fluid.fluid_composition = dict(zip(
                fluid.fluid_object.species_names,  # keys (species)
                fluid.fluid_object.X  # values (molar composition)
            ))
            return fluid

    class AirNASA9:
        """Model of air based on NASA 9-coefficient parameterisation model."""

        @staticmethod
        def Air() -> Fluid:
            """Air with standard composition, temperature, and pressure."""
            mech = "airnasa9.yaml"
            compositionX = {
                "N2": 78.084,
                "O2": 20.946,
            }
            return Fluid.from_cantera_mech(mech, compositionX)


if __name__ == "__main__":
    gas = Fluids.PerfectGas.CarbonDioxide()

    print(gas.gamma)

    gas2 = Fluids.PerfectGas.Hydrogen()

    print(gas2.gamma)
