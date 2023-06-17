"""Module implementing equations of state for gases."""
import numpy as np
import re

from carpy.utility import Quantity, cast2numpy, constants as co
from carpy.gaskinetics._chemistry_molecules import MoleculeAcyclic

__all__ = ["GasModels"]
__author__ = "Yaseen Reza"

# Redesignate
Molecule = MoleculeAcyclic


# ============================================================================ #
# Thermodynamic state variables, Gas mixing
# ---------------------------------------------------------------------------- #
class IntensiveProperties(object):
    """Data class for tracking intensive properties of thermodynamic state."""

    _rho = Quantity(np.nan, "kg m^{-3}")
    _cv = Quantity(np.nan, "J kg^{-1} K^{-1}")
    _u = Quantity(np.nan, "J kg^{-1}")
    _p = Quantity(np.nan, "Pa")
    _T = Quantity(np.nan, "K")

    # -------------------- #
    # Intensive properties #
    # -------------------- #

    @property
    def a(self) -> Quantity:
        """Chemical thermodynamic activity."""
        return NotImplemented

    @property
    def mu(self) -> Quantity:
        """Chemical potential."""
        return NotImplemented

    @property
    def betaT(self) -> Quantity:
        """Isothermal compressibility."""
        return NotImplemented

    @property
    def betaS(self) -> Quantity:
        """Adiabatic compressibility."""
        return NotImplemented

    @property
    def Kf(self) -> Quantity:
        """Cryoscopic constant."""
        return NotImplemented

    @property
    def rho(self) -> Quantity:
        """Density."""
        return self._rho

    @property
    def Kb(self) -> Quantity:
        """Ebullioscopic."""
        return NotImplemented

    @property
    def h(self) -> Quantity:
        """Specific enthalpy."""
        return NotImplemented

    @property
    def s(self) -> Quantity:
        """Specific entropy."""
        # Probably have to work out molecular inertia, and then use:
        # https://doi.org/10.3390/e21050454
        return NotImplemented

    @property
    def f(self) -> Quantity:
        """Fugacity."""
        return NotImplemented

    @property
    def g(self) -> Quantity:
        """Specific Gibbs free energy."""
        return NotImplemented

    @property
    def cp(self) -> Quantity:
        """Specific heat capacity at constant pressure (isobaric)."""
        return NotImplemented

    @property
    def cv(self) -> Quantity:
        """Specific heat capacity at constant volume (isochoric)."""
        return self._cv

    @property
    def u(self) -> Quantity:
        """Specific internal energy."""
        return self._u

    @property
    def piT(self) -> Quantity:
        """Internal pressure."""
        return NotImplemented

    @property
    def p(self) -> Quantity:
        """Pressure."""
        return self._p

    @property
    def T(self) -> Quantity:
        """Temperature."""
        return self._T

    @property
    def k_thermal(self) -> Quantity:
        """Thermal conductivity."""
        return NotImplemented

    @property
    def alpha(self) -> Quantity:
        """Thermal diffusivity."""
        return NotImplemented

    @property
    def alphaV(self) -> Quantity:
        """Volumetric coefficient of thermal expansion at constant pressure."""
        return NotImplemented

    @property
    def Chi(self) -> Quantity:
        """Vapour quality."""
        return NotImplemented

    @property
    def nu(self) -> Quantity:
        """Specific volume."""
        return 1 / self.rho

    @property
    def gamma(self) -> float:
        """Specific heat ratio (a.k.a heat capacity ratio, adiabatic index)."""
        return float(self.cp / self.cv)


class StateVars(IntensiveProperties):
    """
    Data class for tracking intensive, extensive, and other derived properties
    of thermodynamic state in a system.
    """
    _n: Quantity

    # -------------------- #
    # Extensive properties #
    # -------------------- #

    @property
    def H(self) -> Quantity:
        """Enthalpy."""
        return self.m * self.h

    @property
    def S(self) -> Quantity:
        """Entropy."""
        return self.m * self.s

    @property
    def G(self) -> Quantity:
        """Gibbs free energy."""
        return self.m * self.g

    @property
    def Xi(self) -> Quantity:
        """Planck potential, Gibbs free entropy."""
        return self.Phi - self.p * self.V / self.T

    @property
    def Ohm(self) -> Quantity:
        """Landau potential, Landau free energy."""
        return self.U - self.T * self.S - self.mu * self.N

    @property
    def Cp(self) -> Quantity:
        """Heat capacity at constant pressure (isobaric)."""
        return self.m * self.cp

    @property
    def Cv(self) -> Quantity:
        """Heat capacity at constant volume (isochoric)."""
        return self.m * self.cv

    @property
    def F(self) -> Quantity:
        """Helmholtz free energy."""
        return self.U - self.T * self.S

    @property
    def Phi(self) -> Quantity:
        """Massieu potential, Helmholtz free entropy."""
        return self.S - self.U / self.T

    @property
    def U(self) -> Quantity:
        """Internal energy."""
        return self.m * self.u

    @property
    def m(self) -> Quantity:
        """Mass."""
        return NotImplemented

    @property
    def N(self) -> int:
        """Number of particles."""
        return int(self.n * co.PHYSICAL.N_A)

    @property
    def V(self) -> Quantity:
        """Volume."""
        return self.m * self.nu

    @property
    def n(self) -> Quantity:
        """Molar mount of substance."""
        return self._n


class PerfectGases(StateVars):
    """
    A class for modelling mixtures of ideal (perfect or semi-perfect) gases.

    Examples: ::

        # Create a gas approximately representative of the composition of air
        >>> air = PerfectGases().X = "N2:78.0, O2:21.0"

    """
    _X: dict[Molecule, float]  # Template for what composition should look like
    _n = Quantity(1, "mol")  # Define an amount of substance, just for the lulz

    def __init__(self, name: str = None):
        """
        Args:
            name: For your own convenience, a unique identifier for this object.
        """
        self._name = name
        return

    def __repr__(self):
        reprstr = f"{type(self).__name__}(name='{self.name}')"
        return reprstr

    @property
    def name(self) -> str:
        """Human-readable name of the gas composition."""
        if self._name is None:
            return 'unnamed-gas'
        return self._name

    @staticmethod
    def _XYparse(value) -> dict:
        # User has set value as a dictionary
        if isinstance(value, dict):
            out = dict()
            # Make sure dictionary entries are molecule or string objects
            for k, v in value.items():
                if isinstance(k, Molecule):
                    out[k] = float(v)
                elif isinstance(k, str):
                    out[Molecule(k)] = float(v)
                else:
                    errormsg = (
                        f"Expected one type of either 'str' or "
                        f"{Molecule.__name__}, got type={type(k)} from "
                        f"object {k.__repr__()}"
                    )
                    raise TypeError(errormsg)

        # User has set value as a string
        elif isinstance(value, str):
            out = dict([
                (Molecule(formula.replace(" ", "")), float(v))
                for (formula, v) in re.findall(r"(.+?):(\d*\.?\d*),?\s*", value)
            ])
            # Check if user accidentally omitted mole amount on single molecule
            if len(out) == 0:
                try:
                    out = dict([(Molecule(value.replace(" ", "")), 1.0)])
                except Exception:
                    errormsg = f"Couldn't parse gas components in '{value}'"
                    raise ValueError(errormsg)
            # If mole amounts are given, molecules should match number of colons
            elif len(out) < value.count(":"):
                raise ValueError(f"Lossy conversion from '{value}' to '{out}'")

        # Invalid argument
        else:
            errormsg = f"{value} is not a valid definition of gas composition"
            raise ValueError(errormsg)

        # The compositional dictionary is valid, now normalise fraction to unity
        vector_magnitude = sum(out.values())
        out = {k: (v / vector_magnitude) for (k, v) in out.items()}
        return out

    @property
    def X(self) -> dict[Molecule, float]:
        """Composition of gas, by mole fraction X."""
        return self._X

    @X.setter
    def X(self, value):
        self._X = self._XYparse(value=value)
        return

    @property
    def Y(self) -> dict[Molecule, float]:
        """Composition of gas, by mass fraction Y."""
        X_i = cast2numpy(list(self.X.values()))
        M_i = cast2numpy([float(molecule.M) for molecule in self.X.keys()])
        M = np.sum(X_i * M_i)
        Y_i = (X_i * (M_i / M))
        Y = dict(zip(self.X.keys(), Y_i))
        return Y

    @Y.setter
    def Y(self, value):
        Y = self._XYparse(value=value)
        Y_i = cast2numpy(list(Y.values()))
        M_i = cast2numpy([float(molecule.M) for molecule in Y.keys()])
        M = 1 / np.sum(Y_i / M_i)
        X_i = (Y_i / (M_i / M))
        self._X = dict(zip(Y.keys(), X_i))
        return

    @property
    def _M_components(self) -> Quantity:
        """Molar mass of the gas mixture, by gas."""
        X_i = cast2numpy(list(self.X.values()))
        M_i = Quantity([key.M[0] for key in self.X.keys()], units="kg mol^{-1}")
        M_i = (X_i * M_i)
        return M_i

    @property
    def M(self) -> Quantity:
        """Average molar mass of the gas mixture."""
        M = self._M_components.sum()
        return M

    @property
    def m(self) -> Quantity:
        """Mass of gas in the mixture."""
        return self.n * self.M

    @property
    def cp(self) -> Quantity:
        """Specific heat capacity at constant pressure (isobaric)."""
        return self.cv + co.PHYSICAL.R / self.M

    @property
    def Rs(self) -> Quantity:
        """Specific gas constant."""
        Rs = co.PHYSICAL.R / self.M
        return Rs

    @property
    def c_sound(self) -> Quantity:
        """Speed of sound in an ideal gas."""
        # Isentropic bulk modulus (Ks)
        Ks = self.gamma * self.p
        c_sound = (Ks / self.rho) ** 0.5
        return c_sound


# ============================================================================ #
# Gas Modelling with Equations of State
# ---------------------------------------------------------------------------- #
class GasModel(object):
    """
    object: "What is my purpose?"
    yaseen: "You are a typehint."
    object: "oh my god."
    """
    pass


class PerfectCaloric(GasModel, PerfectGases):
    """
    Equation of state model for an ideal, calorically perfect gas.

    Gases modelled as calorically perfect have specific heat capacities that are
    independent of temperature.
    """

    def __init__(self, name: str = None):
        # Superclass call
        PerfectGases.__init__(self, name=name)
        return

    def __copy__(self):
        # Shallow copy - take name and state
        copy = type(self)(name=self._name)
        copy.X = self.X
        copy.state_TP = self.state_TP
        return copy

    def copy(self):
        """Return a shallow copy of the object (a new gas object)."""
        return self.__copy__()

    @property
    def _cv_components(self) -> Quantity:
        """Specific heat capacity at constant volume (isochoric), by gas."""
        molecule_dof = np.zeros(len(self.X))
        for i, molecule in enumerate(self.X.keys()):
            axe_all = molecule.atom_AXE
            dof = 0

            # Translational DoF
            dof += 3
            if len(axe_all) == 0:
                molecule_dof[i] = dof
                continue  # There were no bonds to any other atoms/ligands, skip

            # Rotational DoF
            axe_central = {
                atom: xe for (atom, xe) in axe_all.items()
                if len(xe["X"]) > 1
            }
            linear = all([
                len(v["X"]) == 2 and v["E"] == 0 for v in axe_central.values()])
            dof += 2
            if linear is False:
                dof += 1

            molecule_dof[i] = dof
            continue  # Other degrees of freedom are not considered

        # Compute isochoric heat capacity
        Xi = cast2numpy(list(self.X.values()))
        Yi = cast2numpy(list(self.Y.values()))
        Mi = (Yi / Xi) * self.M
        Ri = Yi * (co.PHYSICAL.R / Mi)
        cv_i = ((molecule_dof / 2) * Ri)
        return cv_i

    @property
    def _cv(self) -> Quantity:
        """Specific heat capacity at constant volume (isochoric)."""
        return self._cv_components.sum()

    @property
    def _u_components(self) -> Quantity:
        """Specific internal energy, by gas."""
        return self._cv_components * self.T

    @property
    def _u(self) -> Quantity:
        """Specific internal energy."""
        return self._u_components.sum()

    @property
    def h(self) -> Quantity:
        """Specific enthalpy."""
        return self.u + self.p * self.nu

    @property
    def state_TP(self) -> tuple[Quantity, Quantity]:
        """Thermodynamic state temperature and pressure variables."""
        return self.T, self.p

    @state_TP.setter
    def state_TP(self, value):
        # Unpack value and recast
        T, p = value
        T = self._T if T is None else Quantity(T, "K")
        p = self._p if p is None else Quantity(p, "Pa")

        # Compute missing parameters
        rho = p / self.Rs / T

        # Make assignments
        self._rho = rho
        self._p = p
        self._T = T
        return

    @property
    def state_TD(self) -> tuple[Quantity, Quantity]:
        """Thermodynamic state temperature and density variables."""
        return self.T, self.rho

    @state_TD.setter
    def state_TD(self, value):
        # Unpack value and recast
        T, rho = value
        T = self._T if T is None else Quantity(T, "K")
        rho = self._rho if rho is None else Quantity(rho, "kg m^{-3}")

        # Compute missing parameters
        p = rho * self.Rs * T

        # Make assignments
        self._rho = rho
        self._p = p
        self._T = T
        return


class PerfectThermic(GasModel, PerfectGases):
    """
    Equation of state model for an ideal, thermally perfect gas.

    Gases modelled as thermally perfect have specific heat capacities that are
    functions of temperature alone.
    """

    def __init__(self, name: str = None):
        # Superclass call
        PerfectGases.__init__(self, name=name)
        raise NotImplementedError("Sorry! Use 'PerfectCaloric' instead")


class GasModels(object):
    """A collection of gas models, packaged for easy access."""

    # Model typehint
    typehint = GasModel

    # Models
    PerfectCaloric = PerfectCaloric


# Destructive instantiation (no one will want the class of this anyway)
GasModels = GasModels()

# # ============================================================================ #
# # Gas Modelling with Equations of State
# # ---------------------------------------------------------------------------- #
# # ---------------------------------------------------------------------------- #
# class ModelRK(BaseGas):
#     """
#     Real gas model based on the Redlich-Kwong equation of state.
#     """
#
#     def __init__(self, formula: str):
#         """
#         Args:
#             formula: Condensed structural formula of the gas being modelled.
#         """
#         super().__init__(formula)
#         return
#
#     def _f_Z(self, T: np.ndarray, p: np.ndarray):
#         # Recast as necessary
#         T, p = np.broadcast_arrays(T, p)
#         critT, critp = self.molecule.critical_point
#         if np.isnan(critT) or np.isnan(critp):
#             raise ValueError("Couldn't find data for critical point.")
#         else:
#             critT = float(critT)
#             critp = float(critp)
#         R = co.PHYSICAL.R.x
#
#         # Molecule-dependent quantities
#         croot2 = 2 ** (1 / 3)
#         a = 1 / (9 * (croot2 - 1)) * R ** 2 * critT ** 2.5 / critp
#         b = (croot2 - 1) / 3 * R * critT / critp
#
#         def solver(temperature: float, pressure: float):
#             """Solve for compressibility coefficient given scalar quantities."""
#
#             # Solve for the molar volume that makes the whole expression work
#             def f_opt(x):
#                 """Equation solving for the molar volume (V/n)."""
#                 lhs = pressure * x / R / temperature
#                 rhs = x / (x - b) - a / (R * temperature ** 1.5 * (x + b))
#                 return lhs - rhs
#
#             V_m = newton(f_opt, 0.024)
#             Z = pressure * V_m / R / temperature
#             return Z
#
#         return np.vectorize(solver)(T, p)
