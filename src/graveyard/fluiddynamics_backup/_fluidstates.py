"""Module for defining the instantaneous state and behaviour of a fluid."""
from typing import Union
import re

import cantera as ct
import numpy as np
import periodictable as pt

from carpy.utility import Hint, Quantity, revert2scalar, constants as co

__all__ = ["FluidState", "FluidProperties"]
__author__ = "Yaseen Reza"


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
            fluid_object = self._parent.fluid_object
            attribute, units = attr_map[item]
            if isinstance(fluid_object, (ct.Solution, ct.PureFluid)):
                return Quantity(getattr(fluid_object, attribute), units)
            elif isinstance(fluid_object, (LinearGases,)):
                nonprivattr = item[1:]
                return getattr(fluid_object, nonprivattr)
            elif isinstance(fluid_object, np.ndarray):
                output = np.zeros(fluid_object.shape)
                output.flat = [
                    getattr(fluid_object.flat[i], attribute)
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

    state: FluidState
    props: FluidProperties
    _fluid_model: Union[ct.Solution, ct.PureFluid] = None
    _fluid_mech: str = None
    _fluid_comp: str = None

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

    def __setattr__(self, name, value):
        # Override more local variables first:
        # Accessing self.state and self.props causes recursion because it's not
        # in dir(self), but actually in __annotations__ - that's why it's here
        if name in dir(self) + list(self.__annotations__):
            super().__setattr__(name, value)
            return None

        # Otherwise, check the FluidState and FluidProperties (if they exist)
        elif name in dir(self.state):
            self.state.__setattr__(name, value)
            return None
        elif name in dir(self.props):
            self.props.__setattr__(name, value)
            return None

        # The parameter does not exist (and shouldn't be set)
        errormsg = (
            f"Setting parameter '{name}' for {type(self).__name__} object "
            f"is disallowed for user safety (prevents recursive getattr calls)."
        )
        raise AttributeError(errormsg)

    def __call__(self, T: Hint.nums = None, p: Hint.nums = None):
        # Recast as necessary
        T = self.state.T if T is None else T
        p = self.state.p if p is None else p
        T, p, gas_obj = np.broadcast_arrays(T, p, self.fluid_object)

        # One fluid can evaluate multiple states by broadcasting Cantera objects
        new_canteras = np.empty(T.shape, dtype=type(self))
        for i in range(new_canteras.size):
            if isinstance(gas_obj.flat[i], ct.Solution):
                new_canteras.flat[i] = ct.Solution(self.fluid_mechanism)
                new_canteras.flat[i].X = self.fluid_composition
            elif isinstance(gas_obj.flat[i], ct.PureFluid):
                new_canteras.flat[i] = type(gas_obj.flat[i])()
            else:
                raise NotImplementedError
            new_canteras.flat[i].TP = T.flat[i], p.flat[i]

        # Create the new fluid, apply broadcasted cantera objects
        new_fluid = Fluid()
        new_fluid._fluid_model = new_canteras
        new_fluid._fluid_mech = self.fluid_mechanism
        new_fluid.fluid_composition = self.fluid_composition
        return new_fluid

    @property
    @revert2scalar
    def fluid_object(self):
        """The gas model from which state/properties update and propagate."""
        return self._fluid_model

    @property
    def fluid_mechanism(self) -> str:
        """The name of mechanism or model type used."""
        return self._fluid_mech

    @property
    def fluid_composition(self) -> str:
        """The molar composition of fluid."""
        return self._fluid_comp

    @fluid_composition.setter
    def fluid_composition(self, value):
        # Derive composition string
        if isinstance(value, str):
            X = value
        elif isinstance(value, dict):
            X = ", ".join([f"{x}:{y}" for (x, y) in value.items()])
        else:
            raise TypeError(f"Expected mol composition str/dict, not '{value}'")

        # Try to assign the composition (reject it if it's invalid for the mech)
        try:
            if isinstance(self.fluid_object, ct.Solution):
                self.fluid_object.X = X
            elif isinstance(self.fluid_object, np.ndarray):
                for i in range(self.fluid_object.size):
                    setattr(self.fluid_object.flat[i], "X", X)
            self._fluid_comp = X
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
        obj._fluid_model = fluid
        # It's important to set a Cantera object's state AFTER its composition
        obj._fluid_model.TP = 288.15, 101325.0
        return obj

    @classmethod
    def from_cantera_mech(cls, mechanism: str,
                          X: Union[dict[str, float], str] = None):
        """
        Create a Fluid object, from a Cantera fluid.

        Args:
            mechanism: Cantera mechanism filename, e.g. "gri30.yaml".
            X: Dictionary or string describing the molar composition of the
                fluid. For example, "CH4:1, O2:2" creates a stoichiometric mix
                of methane and oxygen. Optional, defaults to None (which
                selects a species from the list of species in the mechanism).

        Returns:
            Fluid object.

        """
        obj = cls()
        obj._fluid_model = ct.Solution(mechanism)
        obj._fluid_mech = mechanism
        if X is not None:
            obj.fluid_composition = X
        # It's important to set a Cantera object's state AFTER its composition
        obj._fluid_model.TP = 288.15, 101325.0
        return obj

    @classmethod
    def from_gasmodel_perfect(cls, X: str):
        obj = cls()
        obj._fluid_model = LinearGases(species=X)
        obj._fluid_mech = "PerfectGas"
        obj._fluid_comp = X
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
        # This is the same as cp - cv = R unless gamma has more complexity to it
        return (self.gamma - 1) / self.gamma * self.state.cp

    @property
    def a(self) -> Quantity:
        """Local speed of sound."""
        # By using pressure and specific volume, we automatically account for
        # compressibility factor Z (as in p * v = Z * R * T)
        return (self.gamma * self.p * self.state.v) ** 0.5


re_atomstyle = re.compile(r"[A-Z][a-z]{,2}")
pt_symbols = [x for x in dir(pt) if re_atomstyle.match(x)]
re_atoms = re.compile("|".join(pt_symbols))
re_atomgroups = re.compile(f"({re_atoms.pattern})" + r"([0-9.]*)")


class LinearGases(object):
    """
    Perfect (a.k.a. calorically perfect) gas model, designed to stand in place
    of a Cantera Solution object and support thermodynamic modules of this
    library.
    """
    _W: Quantity
    _X: str

    def __init__(self, species: str):
        self.T = 288.15
        self.P = 101_325.0
        # Invoke special behaviour when X is set...
        self.X = species
        return

    @property
    def T(self) -> Quantity:
        """Temperature."""
        return self._T

    @T.setter
    def T(self, value):
        self._T = Quantity(value, "K")

    @property
    def P(self) -> Quantity:
        """Pressure."""
        return self._p

    @P.setter
    def P(self, value):
        self._p = Quantity(value, "Pa")

    @property
    def TP(self) -> tuple[Quantity, Quantity]:
        """Temperature and pressure."""
        return self.T, self.P

    @property
    def R(self) -> Quantity:
        """Specific gas constant, R."""
        R = co.PHYSICAL.R / self.W
        return R

    @property
    def W(self) -> Quantity:
        """Molecular mass per unit mole."""
        return self._W

    @property
    def X(self) -> str:
        """Molecular species."""
        return self._X

    @X.setter
    def X(self, value):
        # Find pairs of (element, number of atoms)
        atomgroups = re_atomgroups.findall(value)
        atomgroups = [
            (x, float(y)) if y != "" else (x, 1.)  # If not specified, one atom
            for (x, y) in atomgroups
        ]

        # Error handling: complicated molecule
        linear = (N := sum([num for (_, num) in atomgroups])) <= 3
        if not linear or len(atomgroups) > 2:
            errormsg = (
                f"{type(self).__name__}.X was expecting a linear molecule, "
                f"instead got X='{value}'"
            )
            raise ValueError(errormsg)

        # Error handling: find things that look like atoms but actually aren't
        likeatoms = re_atomstyle.findall(value)
        falseatoms = set(likeatoms) - set(x for x, y in atomgroups)
        if falseatoms:
            errormsg = f"Unrecognised elements in '{value}': {falseatoms}"
            raise ValueError(errormsg)

        # Compute the weight of the molecule (kg/kmol)
        W = sum([getattr(pt, elem).mass * num for (elem, num) in atomgroups])
        self._W = Quantity(W, "kg kmol^{-1}")

        # Compute the excited degrees of freedom at reasonable temperatures
        dof = 3  # Translational degrees of freedom
        linear = (N := sum([num for (_, num) in atomgroups])) <= 3
        if linear:
            dof += 2  # Rotational degrees of freedom
            # At standard temperature, vibration is not excited!
            # dof += 2 * (3 * N - 5)  # Vibration
        else:
            dof += 3  # Rotational degrees of freedom
            # At standard temperature, vibration is not excited!
            # dof += 2 * (3 * N - 6)  # Vibration
        if N > 2:
            # Ansatz fix for simple molecules with more than 2 atoms
            dof += 2 * (N - 2)

        self._cv = Quantity(dof / 2 * self.R, "J kg^{-1} K^{-1}")

        # Finally, save the value
        self._X = value

    @property
    def cv(self) -> Quantity:
        return self._cv

    @property
    def cp(self) -> Quantity:
        return self.cv + self.R

    @property
    def gamma(self) -> float:
        return (self.cp / self.cv).x

    @property
    def rho(self) -> Quantity:
        """Gas density."""
        rho = self.P / self.R / self.T
        return rho
