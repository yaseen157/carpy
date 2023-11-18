"""Methods for establishing performance constraints of a vehicle concept."""

# TODO: Work out how to sub. sympy T/W equation with Energy Constraint's values

from typing import Type

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
Constraint = type("Constraint", (object,), {})


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
    def constraints(self) -> tuple[Constraint, ...]:
        """A tuple of vehicle design constraints."""
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        # Make sure we got a constraint, or tuple of constraints
        if isinstance(value, Constraint):
            self._constraints = (value,)
        elif isinstance(value, tuple):
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
q, W2S = sp.symbols("q,W2S")  # Wing-loading, dynamic pressure
V, Vdot, Vdot_n, z, zdot = sp.symbols("V,Vdot,Vdot_n,z,zdot")  # pos/vel/acc
g = sp.symbols("g")  # local acceleration due to gravity
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
del comp_accel, comp_lift, comp_drag, comp_weight, comp_thrust  # clr. namespace


# ============================================================================ #
# Public-facing functions and classes
# ---------------------------------------------------------------------------- #
class EnergyConstraint(Constraint):
    """Vehicle energy constraint/design point."""

    _q = None
    _W2S = None
    _V = None
    _Vdot = None
    _Vdot_n = None
    _z = None
    _zdot = None
    _g = None
    _alpha = None
    _epsilon = None
    _theta = None
    _mu = None

    def __new__(cls, *, q: Hint.nums = None, W2S: Hint.nums = None,
                V: Hint.nums = None, Vdot: Hint.nums = None,
                Vdot_n: Hint.nums = None, z: Hint.nums = None,
                zdot: Hint.nums = None, g: Hint.nums = None,
                alpha: Hint.nums = None, epsilon: Hint.nums = None,
                theta: Hint.nums = None, mu: Hint.nums = None):
        # Recast as necessary
        kwargs = dict([  # MUST contain same keys as the sympy governing eqn.!!
            ("q", q), ("W2S", W2S), ("V", V), ("Vdot", Vdot), ("Vdotn", Vdot_n),
            ("z", z), ("zdot", zdot), ("g", g), ("alpha", alpha),
            ("epsilon", epsilon), ("theta", theta), ("mu", mu)
        ])
        del q, W2S, V, Vdot, Vdot_n, z, zdot, g, alpha, epsilon, theta, mu
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

    @property
    def W2S(self) -> Quantity:
        """Wing loading, W/S."""
        return self._W2S

    @W2S.setter
    def W2S(self, value):
        self._W2S = Quantity(float(value), "Pa")
        return

    @property
    def V(self) -> Quantity:
        """Velocity parallel to the direction of motion, V."""
        return self._V

    @V.setter
    def V(self, value):
        self._V = Quantity(float(value), "m s^{-1}")
        return

    @property
    def Vdot(self) -> Quantity:
        """Acceleration along flight trajectory, Vdot."""
        return self._Vdot

    @Vdot.setter
    def Vdot(self, value):
        self._Vdot = Quantity(float(value), "m s^{-2}")
        return

    @property
    def Vdot_n(self) -> Quantity:
        """Acceleration resulting in circular motion of vehicle, Vdot_n."""
        return self._Vdot_n

    @Vdot_n.setter
    def Vdot_n(self, value):
        self._Vdot_n = Quantity(float(value), "m s^{-2}")
        return

    @property
    def z(self) -> Quantity:
        """Geometric altitude, z."""
        return self._z

    @z.setter
    def z(self, value):
        self._z = Quantity(float(value), "m")
        return

    @property
    def zdot(self) -> Quantity:
        """Rate of change of geometric altitude, z."""
        return self._zdot

    @zdot.setter
    def zdot(self, value):
        self._zdot = Quantity(float(value), "m s^{-1}")
        return

    @property
    def g(self) -> Quantity:
        """Local acceleration due to gravity, g."""
        return self._g

    @g.setter
    def g(self, value):
        self._g = Quantity(float(value), "m s^{-2}")
        return

    @property
    def alpha(self) -> float:
        """Angle of attack, alpha."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = float(value)
        return

    @property
    def epsilon(self) -> float:
        """Thrust setting angle, epsilon."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = float(value)
        return

    @property
    def theta(self) -> float:
        """Aircraft pitch angle, theta."""
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = float(value)
        return

    @property
    def mu(self) -> float:
        """Coefficient of rolling friction that applies during takeoff, mu."""
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = float(value)
        return
