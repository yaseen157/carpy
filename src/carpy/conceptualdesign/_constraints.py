"""Methods for establishing performance constraints of a vehicle concept."""
from typing import Type
import warnings

import numpy as np
import sympy as sp

from carpy.utility import Hint, Quantity, cast2numpy

__all__ = ["EnergyConstraint"]
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
    _eqn: object

    def __call__(self, **kwargs):
        # Do not allow users to __call__ non-existent constraints
        if type(self) is Constraint:
            errormsg = f"__call__ is only supported on children of {Constraint}"
            raise RuntimeError(errormsg)

        # Find the name of all the constraint attributes of this instance
        constraint_names = [
            x for x in dir(type(self))  # Should exist in instance's class
            if isinstance(getattr(type(self), x), property)  # Is a property
        ]
        # Find the values of the constraints that are defined in the instance
        subs_defined = {k: getattr(self, k) for k in constraint_names}
        for (k, v) in kwargs.items():
            if subs_defined[k] is not NotImplemented:  # Constraint is overriden
                warnmsg = f"{k} in {self} was overriden with {v}"
                warnings.warn(message=warnmsg, category=RuntimeWarning)
            subs_defined[k] = v

        # Find the atoms of the governing equation that can be substituted
        eqn_symbols = tuple(filter(
            lambda x: isinstance(x, sp.core.symbol.Symbol),
            getattr(self._eqn, "atoms")()
        ))
        symbols_allowed = {k for k in eqn_symbols if k.name in constraint_names}
        subs = {
            k: float(subs_defined[k.name]) for k in symbols_allowed
            if subs_defined[k.name] is not NotImplemented
        }

        # Make the substitution
        return getattr(self._eqn, "subs")(subs)


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


# ----- Energy constraint -----
T2W, W2S, q = sp.symbols("T2W,W2S,q")  # T/W, Wing-loading, dynamic pressure
V, Vdot, Vdot_n, zdot, g = sp.symbols("V,Vdot,Vdot_n,zdot, g")  # pos/vel/acc
alpha, epsilon, theta, mu = sp.symbols("alpha,epsilon,theta,mu")  # Greek
CL, CD = sp.symbols("C_L,C_D")  # Performance coefficients

# Components of thrust to weight (streamwise)
comp_accel = Vdot / g
comp_lift = q / W2S * CL * mu * sp.cos(alpha) ** 2
comp_drag = q / W2S * CD * (mu / 2 * sp.sin(2 * alpha) - 1)
comp_weight = zdot / V + mu * sp.cos(alpha) * sp.cos(theta)
comp_thrust = sp.cos(alpha + epsilon) * (
        1 + mu * sp.cos(alpha) * sp.tan(alpha + epsilon))
# T/W due to streamwise acceleration
eqn_T2W = (comp_accel - comp_lift - comp_drag + comp_weight) / comp_thrust

# Components of thrust to weight (streamnormal)
comp_accel = Vdot_n / g
comp_lift = q / W2S * CL * (mu / 2 * sp.sin(2 * alpha) + 1)
comp_drag = q / W2S * CD * mu * sp.sin(alpha) ** 2
comp_weight = (1 - (zdot / V) ** 2) ** 0.5 + mu * sp.cos(alpha) * sp.cos(theta)
comp_thrust = sp.sin(alpha + epsilon) * (1 + mu * sp.sin(alpha))

# T/W due to streamwise + streamnormal acceleration
eqn_T2W += (comp_accel - comp_lift - comp_drag + comp_weight) / comp_thrust
eqn_T2W = sp.Eq(T2W, eqn_T2W)
del comp_accel, comp_lift, comp_drag, comp_weight, comp_thrust  # clr. namespace


# ============================================================================ #
# Public-facing functions and classes
# ---------------------------------------------------------------------------- #
class EnergyConstraint(Constraint):
    """Vehicle energy constraint/design point."""

    _eqn = eqn_T2W  # <- Governing equation
    _q: Quantity = NotImplemented
    _W2S: Quantity = NotImplemented
    _V: Quantity = NotImplemented
    _Vdot: Quantity = NotImplemented
    _Vdot_n: Quantity = NotImplemented
    _zdot: Quantity = NotImplemented
    _g: Quantity = NotImplemented
    _alpha: float = NotImplemented
    _epsilon: float = NotImplemented
    _theta: float = NotImplemented
    _mu: float = NotImplemented

    def __new__(cls, *, q: Hint.nums = None, W2S: Hint.nums = None,
                V: Hint.nums = None,
                Vdot: Hint.nums = None, Vdot_n: Hint.nums = None,
                zdot: Hint.nums = None, g: Hint.nums = None,
                alpha: Hint.nums = None, epsilon: Hint.nums = None,
                theta: Hint.nums = None, mu: Hint.nums = None):
        # Recast as necessary
        kwargs = dict([  # MUST contain same keys as the sympy governing eqn.!!
            ("q", q), ("W2S", W2S), ("V", V), ("Vdot", Vdot), ("Vdotn", Vdot_n),
            ("zdot", zdot), ("g", g), ("alpha", alpha),
            ("epsilon", epsilon), ("theta", theta), ("mu", mu)
        ])
        del q, W2S, V, Vdot, Vdot_n, zdot, g, alpha, epsilon, theta, mu
        kwargs = broadcast_kwargs(kwargs)

        # Make as many constraint objects as there are broadcasted arguments
        constraints = []
        for i in range(kwargs["q"].size):
            # Create a new constraint instance based on a parent's __new__
            inst = super(EnergyConstraint, cls).__new__(cls, *tuple(), **dict())
            # Store the keyword arguments that would've instantiated this object
            inst.__kwargs__ = dict()
            # Populate attributes with the kwargs passed to this class' __new__
            for (k, v) in kwargs.items():
                if v.flat[i] is not None:
                    setattr(inst, k, v.flat[i])  # <- set attributes if not None
                inst.__kwargs__[k] = v.flat[i]  # <- record the kwargs, always
            constraints.append(inst)

        # Return a Constraints object to the user
        return Constraints(constraints=constraints)

    @property
    def q(self) -> Quantity:
        """Dynamic pressure, q."""
        return self._q

    @q.setter
    def q(self, value):
        self._q = Quantity(float(value), "Pa")
        return

    @q.deleter
    def q(self):
        self._q = NotImplemented
        return

    @property
    def W2S(self) -> Quantity:
        """Wing loading, W/S."""
        return self._W2S

    @W2S.setter
    def W2S(self, value):
        self._W2S = Quantity(float(value), "Pa")
        return

    @W2S.deleter
    def W2S(self):
        self._W2S = NotImplemented
        return

    @property
    def V(self) -> Quantity:
        """Velocity parallel to the direction of motion, V."""
        return self._V

    @V.setter
    def V(self, value):
        self._V = Quantity(float(value), "m s^{-1}")
        return

    @V.deleter
    def V(self):
        self._V = NotImplemented
        return

    @property
    def Vdot(self) -> Quantity:
        """Acceleration along flight trajectory, Vdot."""
        return self._Vdot

    @Vdot.setter
    def Vdot(self, value):
        self._Vdot = Quantity(float(value), "m s^{-2}")
        return

    @Vdot.deleter
    def Vdot(self):
        self._Vdot = NotImplemented
        return

    @property
    def Vdot_n(self) -> Quantity:
        """Acceleration resulting in circular motion of vehicle, Vdot_n."""
        return self._Vdot_n

    @Vdot_n.setter
    def Vdot_n(self, value):
        self._Vdot_n = Quantity(float(value), "m s^{-2}")
        return

    @Vdot_n.deleter
    def Vdot_n(self):
        self._Vdot_n = NotImplemented
        return

    @property
    def zdot(self) -> Quantity:
        """Rate of change of geometric altitude, zdot."""
        return self._zdot

    @zdot.setter
    def zdot(self, value):
        self._zdot = Quantity(float(value), "m s^{-1}")
        return

    @zdot.deleter
    def zdot(self):
        self._zdot = NotImplemented
        return

    @property
    def g(self) -> Quantity:
        """Local acceleration due to gravity, g."""
        return self._g

    @g.setter
    def g(self, value):
        self._g = Quantity(float(value), "m s^{-2}")
        return

    @g.deleter
    def g(self):
        self._g = NotImplemented
        return

    @property
    def alpha(self) -> float:
        """Angle of attack, alpha."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = float(value)
        return

    @alpha.deleter
    def alpha(self):
        self._alpha = NotImplemented
        return

    @property
    def epsilon(self) -> float:
        """Thrust setting angle, epsilon."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = float(value)
        return

    @epsilon.deleter
    def epsilon(self):
        self._epsilon = NotImplemented
        return

    @property
    def theta(self) -> float:
        """Aircraft pitch angle, theta."""
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = float(value)
        return

    @theta.deleter
    def theta(self):
        self._theta = NotImplemented
        return

    @property
    def mu(self) -> float:
        """Coefficient of rolling friction that applies during takeoff, mu."""
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = float(value)
        return

    @mu.deleter
    def mu(self):
        self._mu = NotImplemented
        return
