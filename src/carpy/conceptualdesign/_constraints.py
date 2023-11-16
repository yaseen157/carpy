"""Methods for establishing performance constraints of a vehicle concept."""

__author__ = "Yaseen Reza"


class Constraint(object):
    """Vehicle constraint/design point."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Sorry, this code isn't ready yet")


class Constraints(object):
    """Packaged constraint objects, to be applied simultaneously."""
    _constraints: tuple[Constraint, ...] = ()

    def __init__(self, constraints):
        self.constraints = constraints
        return

    def __add__(self, other):
        new_constraints = self.constraints + other.constraints
        new_object = type(self)(constraints=new_constraints)
        return new_object

    def __radd__(self, other):
        return self.__add__(other)

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """A tuple of vehicle design constraints."""
        return self.constraints

    @constraints.setter
    def constraints(self, value):
        # Make sure we got a constraint, or tuple of constraints
        if isinstance(value, Constraint):
            self._constraints = (value,)
            return
        elif isinstance(value, tuple):
            if len(set(map(type, value)) - {Constraint}) == 0:
                self._constraints = value
                return

        errormsg = (
            f"value must be of type Constraint or tuple[Constraint, ...] "
            f"(actually got {value, type(value)=})"
        )
        raise TypeError(errormsg)
