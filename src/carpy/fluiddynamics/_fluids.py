"""Module for consistent modelling of fluids, including ideal and real gases."""
from typing import Union
import warnings

import cantera as ct
import numpy as np

from carpy.utility import Hint, Quantity, revert2scalar

__all__ = ["Fluid", "Fluids"]
__author__ = "Yaseen Reza"


# class CanteraFluids:
#     """A collection of fluids, with thermokinetic models (from Cantera)."""
#
#     class PureFluid:
#         """Pure, homogeneous fluids."""
#
#         @staticmethod
#         def CarbonDioxide():
#             """Carbon dioxide at standard temperature and pressure."""
#             gas = ct.CarbonDioxide()
#             gas.TP = 288.15, 101325.0
#             return gas
#
#         CO2 = CarbonDioxide
#
#         @staticmethod
#         def Heptane():
#             """Heptane at standard temperature and pressure."""
#             gas = ct.Heptane()
#             gas.TP = 288.15, 101325.0
#             return gas
#
#         C7H16 = Heptane
#
#         @staticmethod
#         def HFC134a():
#             """HFC134a refrigerant at standard temperature and pressure."""
#             gas = ct.Hfc134a()
#             gas.TP = 288.15, 101325.0
#             return gas
#
#         R134a = HFC134a
#
#         @staticmethod
#         def Hydrogen():
#             """Hydrogen at standard temperature and pressure."""
#             gas = ct.Hydrogen()
#             gas.TP = 288.15, 101325.0
#             return gas
#
#         H2 = Hydrogen
#
#         @staticmethod
#         def Methane():
#             """Methane at standard temperature and pressure."""
#             gas = ct.Methane()
#             gas.TP = 288.15, 101325.0
#             return gas
#
#         CH4 = Methane
#
#         @staticmethod
#         def Nitrogen():
#             """Nitrogen at standard temperature and pressure."""
#             gas = ct.Nitrogen()
#             gas.TP = 288.15, 101325.0
#             return gas
#
#         N2 = Nitrogen
#
#         @staticmethod
#         def Oxygen():
#             """Oxygen at standard temperature and pressure."""
#             gas = ct.Oxygen()
#             gas.TP = 288.15, 101325.0
#             return gas
#
#         O2 = Oxygen
#
#         @staticmethod
#         def Water():
#             """Water at standard temperature and pressure."""
#             gas = ct.Water()
#             gas.TP = 288.15, 101325.0
#             return gas
#
#         H2O = Water
#
#     class GRI30:
#         """Gases based on GRI-Mech 3.0 combustion model."""
#
#         @staticmethod
#         def Air():
#             """Air with standard composition, temperature, and pressure."""
#             gas = ct.Solution("gri30.yaml")
#             compositionX = {
#                 "N2": 78.084,
#                 "O2": 20.946,
#                 "Ar": 0.9340,
#                 "CO2": 0.0407,
#                 "CH4": 0.00018,
#                 "H2": 0.000055
#             }
#             gas.X = ", ".join([f"{x}:{y}" for (x, y) in compositionX.items()])
#             gas.TP = 288.15, 101325.0
#             gas.name = "air_gri30"
#             return gas
#
#     class GRI30highT:
#         """
#         Gases based on GRI-Mech 3.0 combustion model, with high temperature
#         modifications.
#         """
#
#         @staticmethod
#         def Air():
#             """Air with standard composition, temperature, and pressure."""
#             gas = ct.Solution("gri30_highT.yaml")
#             compositionX = {
#                 "N2": 78.084,
#                 "O2": 20.946,
#                 "Ar": 0.9340,
#                 "CO2": 0.0407,
#                 "CH4": 0.00018,
#                 "H2": 0.000055
#             }
#             gas.X = ", ".join([f"{x}:{y}" for (x, y) in compositionX.items()])
#             gas.TP = 288.15, 101325.0
#             gas.name = "air_gri30-highT"
#             return gas
#
#     class Air:
#         """Default air model, as given in Cantera's air.yaml."""
#
#         @staticmethod
#         def Air():
#             """Air with standard composition, temperature, and pressure."""
#             gas = ct.Solution("air.yaml")
#             gas.TP = 288.15, 101325.0
#             gas.name = "air"
#             return gas
#
#     class AirNASA9:
#         """Model of air based on NASA 9-coefficient parameterisation model."""
#
#         @staticmethod
#         def Air():
#             """Air with standard composition, temperature, and pressure."""
#             gas = ct.Solution("airnasa9.yaml")
#             compositionX = {
#                 "N2": 78.084,
#                 "O2": 20.946,
#             }
#             gas.X = ", ".join([f"{x}:{y}" for (x, y) in compositionX.items()])
#             gas.TP = 288.15, 101325.0
#             gas.name = "air_nasa9"
#             return gas


class FluidState(object):
    """
    Thermochemical and kinetic properties of a fluid.
    """
    _a: NotImplemented
    _mu_i: Quantity
    _beta_S: Quantity
    _beta_T: Quantity
    _Kf: Quantity
    _rho: Quantity
    _Kb: Quantity
    _h: Quantity
    _s: Quantity
    _f: Quantity
    _g: Quantity
    _cp: Quantity
    _cv: Quantity
    _u: Quantity
    _pi_T: Quantity
    _p: Quantity
    _T: Quantity
    _k: Quantity
    _alpha: Quantity
    _alpha_L: Quantity
    _alpha_A: Quantity
    _alpha_V: Quantity
    _chi: float
    _v: Quantity

    def __init__(self, parent):
        self._parent = parent

    def __getattr__(self, item):
        # An item that is undefined can probably be found in the Cantera object
        attr_map = {
            # state.a
            # state.mu_i
            # state.beta_S
            "_beta_T": ("isothermal_compressibility", "Pa^{-1}"),
            # state.Kf
            "_rho": ("density", "kg m^{-3}"),
            # state.Kb
            "_h": ("enthalpy_mass", "J kg^{-1}"),
            "_s": ("entropy_mass", "J kg^{-1} K^{-1}"),
            # state.f
            "_g": ("gibbs_mass", "J kg^{-1}"),
            "_cp": ("cp_mass", "J kg^{-1} K^{-1}"),
            "_cv": ("cv_mass", "J kg^{-1} K^{-1}"),
            "_u": ("int_energy_mass", "J kg^{-1}"),
            "_p": ("P", "Pa"),
            "_T": ("T", "K"),
            "_k": ("thermal_conductivity", "W m^{-1} K^{-1}"),
            "_alpha_V": ("thermal_expansion_coeff", "K^{-1}"),
            # state.chi
        }
        if item in attr_map:
            cantera_object = self._parent.cantera_object
            attribute, units = attr_map[item]
            if isinstance(cantera_object, (ct.Solution, ct.PureFluid)):
                return Quantity(getattr(cantera_object, attribute), units)
            elif isinstance(cantera_object, np.ndarray):
                output = np.zeros(cantera_object.shape)
                output.flat = [
                    getattr(cantera_object.flat[i], attribute)
                    for i in range(output.size)
                ]
                return Quantity(output, units)
        elif item == "_alpha":
            return self.k / (self.rho * self.cp)
        elif item == "_alpha_L":
            return self.alpha_V / 3
        elif item == "_alpha_A":
            return self.alpha_L * 2
        elif item == "_v":
            return 1 / self.rho

        return super().__getattribute__(item)

    @property
    def a(self):
        """Activity, a."""
        return self._a

    @a.setter
    def a(self, value):
        self._a = value

    @property
    def mu_i(self) -> Quantity:
        """Chemical potential, mu_i."""
        return self._mu_i

    @mu_i.setter
    def mu_i(self, value):
        self._mu_i = Quantity(value, "kJ mol^{-1}")

    @property
    def beta_S(self) -> Quantity:
        """Compressibility (adiabatic), beta_S."""
        return self._beta_S

    @beta_S.setter
    def beta_S(self, value):
        self._beta_S = Quantity(value, "Pa^{-1}")

    @property
    def beta_T(self) -> Quantity:
        """Compressibility (isothermal), beta_T."""
        return self._beta_T

    @beta_T.setter
    def beta_T(self, value):
        self._beta_T = Quantity(value, "Pa^{-1}")

    @property
    def Kf(self) -> Quantity:
        """Cryoscopic constant, Kf."""
        return self._Kf

    @Kf.setter
    def Kf(self, value):
        self._Kf = Quantity(value, "K kg mol^{-1}")

    @property
    def rho(self) -> Quantity:
        """Density, rho."""
        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho = Quantity(value, "kg m^{-3}")

    @property
    def Kb(self) -> Quantity:
        """Ebullioscopic constant, Kb."""
        return self._Kb

    @Kb.setter
    def Kb(self, value):
        self._Kb = Quantity(value, "K kg mol^{-1}")

    @property
    def h(self) -> Quantity:
        """Specific enthalpy, h."""
        return self._h

    @h.setter
    def h(self, value):
        self._h = Quantity(value, "J kg^{-1}")

    @property
    def s(self) -> Quantity:
        """Specific entropy, s."""
        return self._s

    @s.setter
    def s(self, value):
        self._s = Quantity(value, "J kg^{-1} K^{-1}")

    @property
    def f(self) -> Quantity:
        """Fugacity, f."""
        return self._f

    @f.setter
    def f(self, value):
        self._f = Quantity(value, "N m^{-2}")

    @property
    def g(self) -> Quantity:
        """Specific Gibbs free energy, g."""
        return self._g

    @g.setter
    def g(self, value):
        self._g = Quantity(value, "J kg^{-1}")

    @property
    def cp(self) -> Quantity:
        """Specific heat capacity (isobaric), cp."""
        return self._cp

    @cp.setter
    def cp(self, value):
        self._cp = Quantity(value, "J kg^{-1} K^{-1}")

    @property
    def cv(self) -> Quantity:
        """Specific heat capacity (isochoric), cv."""
        return self._cv

    @cv.setter
    def cv(self, value):
        self._cv = Quantity(value, "J kg^{-1} K^{-1}")

    @property
    def u(self) -> Quantity:
        """Specific internal energy, u."""
        return self._u

    @u.setter
    def u(self, value):
        self._u = Quantity(value, "J kg^{-1}")

    @property
    def pi_T(self) -> Quantity:
        """Internal pressure, pi_T."""
        return self._pi_T

    @pi_T.setter
    def pi_T(self, value):
        self._pi_T = Quantity(value, "Pa")

    @property
    def p(self) -> Quantity:
        """Pressure, p."""
        return self._p

    @p.setter
    def p(self, value):
        self._p = Quantity(value, "Pa")

    @property
    def T(self) -> Quantity:
        """Temperature, T."""
        return self._T

    @T.setter
    def T(self, value):
        self._T = Quantity(value, "K")

    @property
    def k(self) -> Quantity:
        """Thermal conductivity, k."""
        return self._k

    @k.setter
    def k(self, value):
        self._k = Quantity(value, "W m^{-1} K^{-1}")

    @property
    def alpha(self) -> Quantity:
        """Thermal diffusivity, alpha."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = Quantity(value, "m^{2} s^{-1}")

    @property
    def alpha_L(self) -> Quantity:
        """Thermal expansion (linear), alpha_L."""
        return self._alpha_L

    @alpha_L.setter
    def alpha_L(self, value):
        self._alpha_L = Quantity(value, "K^{-1}")

    @property
    def alpha_A(self) -> Quantity:
        """Thermal expansion (area), alpha_A."""
        return self._alpha_A

    @alpha_A.setter
    def alpha_A(self, value):
        self._alpha_A = Quantity(value, "K^{-1}")

    @property
    def alpha_V(self) -> Quantity:
        """Thermal expansion (volumetric), alpha_V."""
        return self._alpha_V

    @alpha_V.setter
    def alpha_V(self, value):
        self._alpha_V = Quantity(value, "K^{-1}")

    @property
    def chi(self) -> float:
        """Vapour quality, chi."""
        return self._chi

    @chi.setter
    def chi(self, value):
        self._chi = value

    @property
    def v(self) -> Quantity:
        """Specific volume, v."""
        return self._v

    @v.setter
    def v(self, value):
        self._v = Quantity(value, "m^{3} kg^{-1}")


class FluidProperties(object):
    """Kinetic properties of fluids derived from intermolecular interactions."""
    _mu: Quantity
    _lamda: Quantity
    _D: Quantity

    def __init__(self, parent):
        self._parent = parent

    @property
    def mu(self) -> Quantity:
        """Fluid dynamic viscosity."""
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = Quantity(value, "Pa s")

    @property
    def nu(self) -> Quantity:
        """Fluid kinematic viscosity."""
        return self.mu / self._parent.state.rho

    @nu.setter
    def nu(self, value):
        self.mu = (value * self._parent.state.rho).x

    @property
    def lamda(self) -> Quantity:
        """Mean molecular free path distance."""
        return self._lamda

    @lamda.setter
    def lamda(self, value):
        self._lamda = Quantity(value, "m")

    @property
    def D(self) -> Quantity:
        """Mass diffusivity (coefficient of mass transport by diffusion)."""
        return self._D

    @D.setter
    def D(self, value):
        self._D = Quantity(value, "m^{2} s^{-1}")


class Fluid(object):
    """An object for storing the properties of a fluid."""

    _ct_object: Union[ct.Solution, ct.PureFluid] = None
    _ct_mechanism: str = None
    _ct_composition: str = None

    def __init__(self):
        self.state = FluidState(parent=self)  # Define thermodynamic state
        self.props = FluidProperties(parent=self)  # Define fluid properties
        return

    def __getattr__(self, item):
        # Search for missing attribute in fluid.states and fluid.props
        if item in dir(self.state):
            return getattr(self.state, item)
        if item in dir(self.props):
            return getattr(self.props, item)
        return super().__getattribute__(item)

    def __call__(self, T: Hint.nums = None, p: Hint.nums = None):
        # Recast as necessary
        T = self.state.T if T is None else T
        p = self.state.p if p is None else p
        T, p, ct_obj = np.broadcast_arrays(T, p, self.cantera_object)

        # One fluid can evaluate multiple states by broadcasting Cantera objects
        new_canteras = np.empty(T.shape, dtype=type(self))
        for i in range(new_canteras.size):
            if isinstance(ct_obj.flat[i], ct.Solution):
                new_canteras.flat[i] = ct.Solution(self.cantera_mechanism)
                new_canteras.flat[i].X = self.cantera_composition
            elif isinstance(ct_obj.flat[i], ct.PureFluid):
                new_canteras.flat[i] = type(ct_obj.flat[i])()
            else:
                raise NotImplementedError
            new_canteras.flat[i].TP = T.flat[i], p.flat[i]

        # Create the new fluid, apply broadcasted cantera objects
        new_fluid = Fluid()
        new_fluid._ct_object = new_canteras
        new_fluid._ct_mechanism = self.cantera_mechanism
        new_fluid.cantera_composition = self.cantera_composition
        return new_fluid

    @property
    @revert2scalar
    def cantera_object(self) -> Union[ct.Solution, ct.PureFluid]:
        """A Cantera phase object."""
        return self._ct_object

    @property
    def cantera_mechanism(self) -> str:
        """The name of the Cantera chemical kinetics file used."""
        return self._ct_mechanism

    @property
    def cantera_composition(self) -> str:
        """The molar composition of the Cantera phase object."""
        return self._ct_composition

    @cantera_composition.setter
    def cantera_composition(self, value):
        # Derive composition string
        if isinstance(value, str):
            X = value
        elif isinstance(value, dict):
            X = ", ".join([f"{x}:{y}" for (x, y) in value.items()])
        else:
            raise TypeError(f"Expected mol composition str/dict, not '{value}'")

        try:
            if isinstance(self.cantera_object, ct.Solution):
                self.cantera_object.X = X
            elif isinstance(self.cantera_object, np.ndarray):
                for i in range(self.cantera_object.size):
                    setattr(self.cantera_object.flat[i], "X", X)
            self._ct_composition = X
        except Exception as e:
            raise UserWarning("Cannot set composition") from e

    @classmethod
    def from_cantera_fluid(cls, fluid: Union[ct.Solution, ct.PureFluid]):
        """
        Create a Fluid object, from a Cantera fluid.

        Args:
            fluid: Cantera Solution or PureFluid.

        Returns:
            Fluid object.

        """
        obj = cls()
        obj._ct_object = fluid
        # It's important to set a Cantera object's state AFTER its composition
        obj._ct_object.TP = 288.15, 101325.0
        return obj

    @classmethod
    def from_cantera_mech(cls, mechanism: str,
                          X: Union[dict[str, float], str]):
        """
        Create a Fluid object, from a Cantera fluid.

        Args:
            mechanism: Cantera mechanism filename, e.g. "gri30.yaml".
            X: Dictionary or string describing the molar composition of the
                fluid. For example, "CH4:1, O2:2" creates a stoichiometric mix
                of methane and oxygen.

        Returns:
            Fluid object.

        """
        obj = cls()
        obj._ct_object = ct.Solution(mechanism)
        obj._ct_mechanism = mechanism
        obj.cantera_composition = X
        # It's important to set a Cantera object's state AFTER its composition
        obj._ct_object.TP = 288.15, 101325.0
        return obj

    @property
    def gamma(self) -> float:
        """
        Ratio of specific heats (a.k.a. adiabatic index) for an ideal gas.

        A measure of the "performance" of a gas in expansions, compression, etc.

        Typical values:
        -   Monotomic gases (He, Ar, Ne, etc.):
            >> 1.67
        -   Diatomic gases (H2, N2, O2, etc.):
            >> 1.40 at low/moderate temperature.
            >> 1.29 at moderately high temperature (1000-2000 K).
        -   Triatomic gases (e.g. CO2):
            >> 1.33 at low/moderate temperature.
        -   Polyatomic gases:
            >> lim(gamma) = 1.00+
        """
        return (self.state.cp / self.state.cv).x

    @property
    def gamma_pv(self) -> float:
        """
        Isentropic exponent for a real gas, satisfying the following:

        P * v ** (gamma_pv) = constant.

        Returns:
            gamma_pv.

        """
        dpdv_isothermal = -1 / (self.state.beta_T * self.state.v)
        gamma_pv = -self.state.v / self.state.p * self.gamma * dpdv_isothermal
        return gamma_pv.x

    @property
    def gamma_Tv(self) -> float:
        """
        Isentropic exponent for a real gas, satisfying the following:

        T * v ** (gamma_Tv - 1) = constant.

        Returns:
            gamma_Tv.

        """
        dpdT_isochoric = self.R / self.v  # Ideal gas only...
        gamma_Tv = 1 + self.v / self.cv * dpdT_isochoric
        return gamma_Tv.x

    @property
    def gamma_pT(self) -> float:
        """
        Isentropic exponent for a real gas, satisfying the following:

        T * p ** ((1 - gamma_pT) / gamma_pT) = constant.

        Returns:
            gamma_pT.

        """
        dvdT_isobaric = self.R / self.p  # Ideal gas only...
        gamma_pT = 1 / (1 - self.p / self.cp * dvdT_isobaric)
        return gamma_pT.x

    @property
    def R(self) -> Quantity:
        """
        Specific gas constant.

        The amount of mechanical work obtained by heating a unit mass of the gas
        through a unit temperature rise at constant pressure.
        """
        return (self.gamma - 1) / self.gamma * self.state.cp

    @property
    def a(self) -> Quantity:
        """Local speed of sound."""
        # By using pressure and specific volume, we automatically account for
        # compressibility factor Z (as in p * v = Z * R * T)
        return (self.gamma * self.p * self.state.v) ** 0.5


class Fluids(object):
    """A collection of FLuid objects."""

    class PerfectGas:
        """Perfect (a.k.a. calorically perfect) gases."""

        @staticmethod
        def Monatomic() -> Fluid:
            """Perfect monatomic gas."""
            fluid = Fluid()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fluid.state.gamma = 5 / 3
            return fluid

        @staticmethod
        def Diatomic() -> Fluid:
            """Perfect diatomic gas."""
            fluid = Fluid()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fluid.state.gamma = 7 / 5
            return fluid

        @staticmethod
        def Triatomic() -> Fluid:
            """Perfect triatomic (linear) gas."""
            fluid = Fluid()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fluid.state.gamma = (9 - 1) / (7 - 1)
            return fluid

        @staticmethod
        def TrigonalPlanar() -> Fluid:
            """Perfect trigonal planar gas."""
            fluid = Fluid()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fluid.state.gamma = 9 / 7
            return fluid

    class PureFluid:
        """Pure, homogeneous fluids."""

        @staticmethod
        def CarbonDioxide() -> Fluid:
            """Carbon dioxide at standard temperature and pressure."""
            fluid = ct.CarbonDioxide()
            return Fluid().from_cantera_fluid(fluid)

        CO2 = CarbonDioxide

        @staticmethod
        def Heptane() -> Fluid:
            """Heptane at standard temperature and pressure."""
            fluid = ct.Heptane()
            return Fluid().from_cantera_fluid(fluid)

        C7H16 = Heptane

        @staticmethod
        def HFC134a() -> Fluid:
            """HFC134a refrigerant at standard temperature and pressure."""
            fluid = ct.Hfc134a()
            return Fluid().from_cantera_fluid(fluid)

        R134a = HFC134a

        @staticmethod
        def Hydrogen() -> Fluid:
            """Hydrogen at standard temperature and pressure."""
            fluid = ct.Hydrogen()
            return Fluid().from_cantera_fluid(fluid)

        H2 = Hydrogen

        @staticmethod
        def Methane() -> Fluid:
            """Methane at standard temperature and pressure."""
            fluid = ct.Methane()
            return Fluid().from_cantera_fluid(fluid)

        CH4 = Methane

        @staticmethod
        def Nitrogen() -> Fluid:
            """Nitrogen at standard temperature and pressure."""
            fluid = ct.Nitrogen()
            return Fluid().from_cantera_fluid(fluid)

        N2 = Nitrogen

        @staticmethod
        def Oxygen() -> Fluid:
            """Oxygen at standard temperature and pressure."""
            fluid = ct.Oxygen()
            return Fluid().from_cantera_fluid(fluid)

        O2 = Oxygen

        @staticmethod
        def Water() -> Fluid:
            """Water at standard temperature and pressure."""
            fluid = ct.Water()
            return Fluid().from_cantera_fluid(fluid)

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
            return Fluid.from_cantera_mech(mech)

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
