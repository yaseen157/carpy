"""Methods for establishing performance constraints of a vehicle concept."""
from functools import cache
from typing import Type

import numpy as np
import sympy as sp

from carpy.utility import Quantity, cast2numpy

__all__ = ["Manoeuvre3DTW", "Manoeuvre3DPW"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Support functions and classes
# ---------------------------------------------------------------------------- #

def broadcast_kwargs(kwargs: dict):
    """Checks if inputs of kwargs are broadcastable against each other."""
    # Try broadcasting
    try:
        broadcasted_arr = np.broadcast_arrays(*kwargs.values())
    except ValueError as e:
        errormsg = (
            f"Couldn't broadcast values of parameters with the following "
            f"keys: {[k for (k, v) in kwargs.items() if v is not None]}"
        )
        raise ValueError(errormsg) from e

    # Re-create the input dictionary, but with broadcasted values
    broadcasted = dict(zip(kwargs, map(cast2numpy, broadcasted_arr)))

    return broadcasted


# Generic type for subclasses of constraints
class Constraint(object):
    """Methods for a single constraint."""
    _equation: sp.core.relational.Equality

    @classmethod
    def symbol_map(cls) -> dict[str, sp.core.symbol.Symbol]:
        """
        Maps plaintext symbols to the corresponding symbol object in the
        governing equation.
        """
        symbols = cls._equation.free_symbols
        symbol_mapping = {
            getattr(symbol, "name").replace("C_", "C"): symbol
            for symbol in symbols
        }
        return symbol_mapping

    def __new__(cls, **kwargs):
        # Do not allow users to use Constraint class directly
        if cls is Constraint:
            errormsg = f"__new__ is only supported on children of {Constraint}"
            raise RuntimeError(errormsg)

        # Recast as necessary
        kwargs = {name: kwargs.get(name) for name in cls.symbol_map()}
        kwargs = broadcast_kwargs(kwargs)

        # Make as many constraint objects as there are broadcasted arguments
        constraints_objects = []
        for i in range(kwargs["q"].size):
            # Create a new constraint instance based on a parent's __new__
            inst = super(Constraint, cls).__new__(cls, *(), **{})
            # Store the keyword arguments that would've instantiated this object
            inst.__kwargs__ = dict()
            # Populate attributes with the kwargs passed to this class' __new__
            for (k, v) in kwargs.items():
                if v.flat[i] is not None:
                    setattr(inst, k, v.flat[i])  # <- set attributes if not None
                inst.__kwargs__[k] = v.flat[i]  # <- record the kwargs, always
            constraints_objects.append(inst)

        # Return a Constraints object to the user
        return Constraints(constraints_objects)

    def __call__(self, **kwargs):
        # Do not allow users to use Constraint class directly
        if type(self) is Constraint:
            errormsg = f"__call__ is only supported on children of {Constraint}"
            raise RuntimeError(errormsg)

        # Extract governing equation substitutions from self attributes & kwargs
        symbol_map = self.symbol_map()
        substitutions = broadcast_kwargs({
            k: kwargs.get(k, getattr(self, k))
            for k, v in self.symbol_map().items()
        })

        # Compute outputs
        output = np.empty(list(substitutions.values())[0].shape, dtype=object)
        for i in range(len(output.flat)):
            subs = {
                symbol_map[k]: v.flat[i] for (k, v) in substitutions.items()
                if v.flat[i] is not None
            }
            output.flat[i] = self._equation.subs(subs)

        # Squash output if necessary
        if output.size == 1:
            return output[0]
        return output

    @property
    def P(self) -> Quantity:
        """Propulsive power, P."""
        return self.__kwargs__.get("P", None)

    @P.setter
    def P(self, value):
        self.__kwargs__["P"] = Quantity(float(value), "W")
        return

    @P.deleter
    def P(self):
        del self.__kwargs__["P"]
        return

    @property
    def S(self) -> Quantity:
        """Reference planform wing area, S."""
        return self.__kwargs__.get("S", None)

    @S.setter
    def S(self, value):
        self.__kwargs__["S"] = Quantity(float(value), "m^{2}")
        return

    @S.deleter
    def S(self):
        del self.__kwargs__["S"]
        return

    @property
    def T(self) -> Quantity:
        """Propuslive thrust force, T."""
        return self.__kwargs__.get("T", None)

    @T.setter
    def T(self, value):
        self.__kwargs__["T"] = Quantity(float(value), "N")
        return

    @T.deleter
    def T(self):
        del self.__kwargs__["T"]
        return

    @property
    def V(self) -> Quantity:
        """Velocity parallel to the direction of motion, V."""
        return self.__kwargs__.get("V", None)

    @V.setter
    def V(self, value):
        self.__kwargs__["V"] = Quantity(float(value), "m s^{-1}")
        return

    @V.deleter
    def V(self):
        del self.__kwargs__["V"]
        return

    @property
    def Vdot(self) -> Quantity:
        """Acceleration along flight trajectory, Vdot."""
        return self.__kwargs__.get("Vdot", None)

    @Vdot.setter
    def Vdot(self, value):
        self.__kwargs__["Vdot"] = Quantity(float(value), "m s^{-2}")
        return

    @Vdot.deleter
    def Vdot(self):
        del self.__kwargs__["Vdot"]
        return

    @property
    def Vdot_n(self) -> Quantity:
        """Acceleration resulting in circular motion of vehicle, Vdot_n."""
        return self.__kwargs__.get("Vdot_n", None)

    @Vdot_n.setter
    def Vdot_n(self, value):
        self.__kwargs__["Vdot_n"] = Quantity(float(value), "m s^{-2}")
        return

    @Vdot_n.deleter
    def Vdot_n(self):
        del self.__kwargs__["Vdot_n"]
        return

    @property
    def W(self) -> Quantity:
        """Weight force, W."""
        return self.__kwargs__.get("W", None)

    @W.setter
    def W(self, value):
        self.__kwargs__["W"] = Quantity(float(value), "N")
        return

    @W.deleter
    def W(self):
        del self.__kwargs__["W"]
        return

    @property
    def g(self) -> Quantity:
        """Local acceleration due to gravity, g."""
        return self.__kwargs__.get("g", None)

    @g.setter
    def g(self, value):
        self.__kwargs__["g"] = Quantity(float(value), "m s^{-2}")
        return

    @g.deleter
    def g(self):
        del self.__kwargs__["g"]
        return

    @property
    def q(self) -> Quantity:
        """Dynamic pressure, q."""
        return self.__kwargs__.get("q", None)

    @q.setter
    def q(self, value):
        self.__kwargs__["q"] = Quantity(float(value), "Pa")
        return

    @q.deleter
    def q(self):
        del self.__kwargs__["q"]
        return

    @property
    def zdot(self) -> Quantity:
        """Rate of change of geometric altitude, zdot."""
        return self.__kwargs__.get("zdot", None)

    @zdot.setter
    def zdot(self, value):
        self.__kwargs__["zdot"] = Quantity(float(value), "m s^{-1}")
        return

    @zdot.deleter
    def zdot(self):
        del self.__kwargs__["zdot"]
        return

    @property
    def zddot(self) -> Quantity:
        """Rate of change of geometric climbrate, zddot."""
        return self.__kwargs__.get("zddot", None)

    @zddot.setter
    def zddot(self, value):
        self.__kwargs__["zddot"] = Quantity(float(value), "m s^{-2}")
        return

    @zddot.deleter
    def zddot(self):
        del self.__kwargs__["zddot"]
        return

    @property
    def alpha(self) -> float:
        """Angle of attack, alpha."""
        return self.__kwargs__.get("alpha", None)

    @alpha.setter
    def alpha(self, value):
        self.__kwargs__["alpha"] = float(value)
        return

    @alpha.deleter
    def alpha(self):
        del self.__kwargs__["alpha"]
        return

    @property
    def epsilon(self) -> float:
        """Thrust setting angle, epsilon."""
        return self.__kwargs__.get("epsilon", None)

    @epsilon.setter
    def epsilon(self, value):
        self.__kwargs__["epsilon"] = float(value)
        return

    @epsilon.deleter
    def epsilon(self):
        del self.__kwargs__["epsilon"]
        return

    @property
    def theta(self) -> float:
        """Aircraft pitch angle, theta."""
        return self.__kwargs__.get("theta", None)

    @theta.setter
    def theta(self, value):
        self.__kwargs__["theta"] = float(value)
        return

    @theta.deleter
    def theta(self):
        del self.__kwargs__["theta"]
        return

    @property
    def mu(self) -> float:
        """Coefficient of rolling friction that applies during takeoff, mu."""
        return self.__kwargs__.get("mu", None)

    @mu.setter
    def mu(self, value):
        self.__kwargs__["mu"] = float(value)
        return

    @mu.deleter
    def mu(self):
        del self.__kwargs__["mu"]
        return

    @property
    def CL(self) -> float:
        """Coefficient of Lift, CL."""
        return self.__kwargs__.get("CL", None)

    @CL.setter
    def CL(self, value):
        self.__kwargs__["CL"] = float(value)
        return

    @CL.deleter
    def CL(self):
        del self.__kwargs__["CL"]
        return

    @property
    def CD(self) -> float:
        """Coefficient of Drag, CD."""
        return self.__kwargs__.get("CD", None)

    @CD.setter
    def CD(self, value):
        self.__kwargs__["CD"] = float(value)
        return

    @CD.deleter
    def CD(self):
        del self.__kwargs__["CD"]
        return


class Constraints(object):
    """Packaged constraint objects, to be applied to a vehicle concept."""
    _constraints: tuple[Type[Constraint], ...] = ()

    def __init__(self, constraints):
        self.constraints = tuple(constraints)
        return

    def __add__(self, other):
        new_constraints = set(self.constraints + other.constraints)
        new_object = type(self)(constraints=new_constraints)
        return new_object

    def __radd__(self, other):
        return self.__add__(other)

    @property
    def constraints(self) -> tuple[Type[Constraint], ...]:
        """A tuple of vehicle design constraints."""
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        # Recast as necessary
        if not isinstance(value, tuple):
            value = (value,)

        # Make sure the tuple contains constraints
        if isinstance(value, tuple):
            # Check that Constraint is the parents of all values in the tuple
            if all([Constraint in x.__bases__ for x in map(type, value)]):
                self._constraints = value
        else:
            errormsg = (
                f"value must be of type Constraint or tuple[Constraint, ...] "
                f"(actually got {value, type(value)=})"
            )
            raise TypeError(errormsg)
        return


class GovEqn(object):
    """Governing equations for constraints analysis tasks."""

    @staticmethod
    @cache
    def TW_3D_ManoeuvringFlight() -> sp.core.relational.Equality:
        """
        Thrust-to-Weight constraint in 3D for Manoeuvring Flight.

        Returns:
            A SymPy "equality" relational object.

        """
        # In-plane
        T, W, S, q, Vdot, g, zdot, V = sp.symbols("T,W,S,q,Vdot,g,zdot,V")
        CD = sp.symbols("C_D")
        alpha, epsilon = sp.symbols("alpha,epsilon")
        sin_g = zdot / V
        eqn_t2w_p = (q / (W / S) * CD + Vdot / g + sin_g) / sp.cos(
            alpha + epsilon)

        # Normal
        zddot, Vdot_n = sp.symbols("zddot, Vdot_n")
        CL = sp.symbols("C_L")
        cos_g = sp.sqrt(1 - sin_g ** 2)
        eqn_t2w_n = (sp.sqrt(
            ((zddot - Vdot * sin_g) / g / cos_g + cos_g) ** 2
            + (Vdot_n / g) ** 2
        ) - (q / (W / S) * CL)) / sp.sin(alpha + epsilon)

        # Combined
        eqn_t2w = sp.Eq(T / W, eqn_t2w_p + eqn_t2w_n)
        return eqn_t2w

    @classmethod
    def PW_3D_ManoeuvringFlight(cls) -> sp.core.relational.Equality:
        """
        Power-to-Weight constraint in 3D for Manoeuvring Flight.

        Returns:
            A SymPy "equality" relational object.

        """
        eqn_t2w = cls.TW_3D_ManoeuvringFlight()

        P, V, W = sp.symbols("P,V,W")

        eqn_p2w = sp.Eq(P / W, eqn_t2w.rhs * V)

        return eqn_p2w

    @staticmethod
    @cache
    def TW_2D_Takeoff() -> sp.core.relational.Equality:
        """
        Thrust-to-Weight constraint in 2D for Takeoff.

        Returns:
            A SymPy "equality" relational object.

        """
        # In-plane
        T, W, Vdot, g, q, S, zdot, V = sp.symbols("T,W,Vdot,g,q,S,zdot,V")
        CL, CD = sp.symbols("C_L,C_D")
        alpha, epsilon, theta, mu = sp.symbols("alpha,epsilon,theta,mu")
        sin_a, cos_a = sp.sin(alpha), sp.cos(alpha)
        sin_g = zdot / V
        factor = mu * sp.cos(alpha) + sp.sin(alpha)
        term1 = CL * (factor * cos_a)
        term2 = CD * (factor * sin_a - 1)
        term3 = sin_g + factor * sp.cos(theta)
        term4 = sp.cos(alpha + epsilon) + factor * sp.sin(epsilon)
        eqn_t2w_p = (Vdot / g - q / (W / S) * (term1 + term2) + term3) / term4

        # Normal
        Vdot_n = sp.symbols("Vdot_n")
        factor = mu * sp.sin(alpha) - sp.cos(alpha)
        cos_g = sp.sqrt(1 - sin_g ** 2)
        term1 = CL * (1 + factor * sp.cos(alpha))
        term2 = CD * factor * sp.sin(alpha)
        term3 = cos_g + factor * sp.cos(theta)
        term4 = sp.sin(alpha + epsilon) + factor * sp.sin(epsilon)
        eqn_t2w_n = (Vdot_n / g - q / (W / S) * (term1 + term2) + term3) / term4

        # Combined
        eqn_t2w = sp.Eq(T / W, eqn_t2w_p + eqn_t2w_n)
        return eqn_t2w


# ============================================================================ #
# Public-facing functions and classes
# ---------------------------------------------------------------------------- #
class Manoeuvre3DTW(Constraint):
    """
    Vehicle energy constraint/design point, for optimising vehicle thrust.
    """
    _equation = GovEqn.TW_3D_ManoeuvringFlight()  # <- Governing equation


class Manoeuvre3DPW(Constraint):
    """
    Vehicle energy constraint/design point, for optimising vehicle power.
    """
    _equation = GovEqn.PW_3D_ManoeuvringFlight()  # <- Governing equation
