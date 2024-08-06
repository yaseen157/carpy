"""A module of maths utilities."""
import numpy as np

from carpy.utility._miscellaneous import broadcast_vector, NumberSets

__all__ = ["RationalNumber", "gradient1d"]
__author__ = "Yaseen Reza"


class RationalNumber:
    _p: int
    _q: int

    def __new__(cls, p, q=None):
        # Recast
        q = 1 if q is None else q

        if p == 0:
            return 0
        elif NumberSets.is_integer(p) and abs(q) == 1:
            return int(p / q)

        obj = super(RationalNumber, cls).__new__(cls)
        return obj

    def __init__(self, p: int | float, q: int | float = None):
        self._p = p  # p can be pos or neg
        self.q = q if q is not None else 1.0  # once q uses property setter, p and q rationalise
        return

    def __repr__(self):
        if abs(self.q) != 1:
            rtn_str = f"{self.p}/{self.q}"
        else:
            rtn_str = f"{self.p}"
        return rtn_str

    def _simplify(self):
        # Rationalise p and q as whole numbers
        log10_p = np.log10(abs(self.p))  # only p can be negative!
        log10_q = np.log10(self.q)
        exp10 = 1 - min(np.floor([log10_p, log10_q]))
        p = int(self.p * 10 ** exp10)
        q = int(self.q * 10 ** exp10)

        # Simplify by greatest common divisor
        gcd = np.gcd(p, q)
        self._p = int(p / gcd)
        self._q = int(q / gcd)
        return

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._p = value
        self._simplify()

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if value < 0:
            self._p = -self._p
            self._q = -value
        elif value == 0:
            error_msg = f"Cannot set the divisor of a {type(self).__name__} type object to zero"
            raise ValueError(error_msg)
        else:
            self._q = value
        self._simplify()

    # ==========================
    # MAGIC MATHEMATICAL METHODS
    # --------------------------
    def __abs__(self):
        """Absolute value."""
        cls = type(self)
        obj = cls(p=abs(self.p), q=self.q)
        return obj

    def __add__(self, other):
        """Addition."""
        cls = type(self)
        if isinstance(other, cls):
            num = self.p * other.q + other.p * self.q
            den = self.q * other.q
            obj = cls(p=num, q=den)
        else:
            obj = cls(p=(self.p + self.q * other), q=self.q)
        return obj

    def __ceil__(self):
        """Ceiling function."""
        return np.ceil(float(self))

    def __divmod__(self, other):
        """Division quotient, remainder."""
        if isinstance(other, type(self)):
            return divmod(float(self), float(other))
        return divmod(float(self), other)

    def __eq__(self, other):
        """Equality."""
        if isinstance(other, type(self)):
            if (self.p == other.p) and (self.q == other.q):
                return True
            return False
        return float(self) == other

    def __float__(self):
        """Cast as float."""
        return self.p / self.q

    def __floor__(self):
        """Floor function."""
        return np.floor(float(self))

    def __floordiv__(self, other):
        """self // other, floored division quotient."""
        return divmod(self, other)[0]

    def __ge__(self, other):
        """Greater than or equal to."""
        return float(self) >= other

    def __gt__(self, other):
        """Greater than."""
        return float(self) > other

    def __int__(self):
        """Cast as integer."""
        return int(float(self))

    def __le__(self, other):
        """Less than or equal to."""
        return float(self) <= other

    def __lt__(self, other):
        """Less than."""
        return float(self) < other

    def __mod__(self, other):
        """Modulo."""
        return divmod(self, other)[1]

    def __mul__(self, other):
        """Multiplication."""
        cls = type(self)
        if isinstance(other, cls):
            num = self.p * other.p
            den = self.q * other.q
            obj = cls(p=num, q=den)
        else:
            obj = cls(p=self.p * other, q=self.q)
        return obj

    def __neg__(self):
        """Negation"""
        cls = type(self)
        obj = cls(p=-self.p, q=self.q)
        return obj

    def __pow__(self, power):
        """self raised to a power"""
        cls = type(self)
        if isinstance(power, cls) and NumberSets.is_integer(power):
            if power >= 0:
                obj = cls(p=self.p ** power.p, q=self.q ** power.p)
            else:
                obj = cls(p=self.q ** -power.p, q=self.p ** -power.p)
        else:
            obj = self.p ** float(power) / (self.q ** float(power))
        return obj

    def __radd__(self, other):
        """Reverse addition."""
        return self.__add__(other)

    def __rdivmod__(self, other):
        """Reverse divmod."""
        return self.__divmod__(other)

    def __rfloordiv__(self, other):
        """Reverse floor division."""
        return self.__floordiv__(other)

    def __rmod__(self, other):
        """Reverse modulo."""
        return self.__rmod__(other)

    def __rmul__(self, other):
        """Reverse multiplication."""
        return self.__mul__(other)

    def __round__(self, n=None):
        """Round function."""
        return round(float(self), ndigits=n)

    def __rpow__(self, other):
        """Reverse raise power."""
        return self.__pow__(other)

    def __rsub__(self, other):
        """Reverse subtraction."""
        return self.__sub__(other)

    def __rtruediv__(self, other):
        """Reverse true division."""
        return self.__truediv__(other)

    def __sub__(self, other):
        """Subtraction."""
        obj = self + (-1 * other)
        return obj

    def __truediv__(self, other):
        """True division."""
        cls = type(self)
        if isinstance(other, cls):
            num = self.p * other.q
            den = self.q * other.p
            obj = cls(p=num, q=den)
        else:
            obj = cls(p=self.p / other, q=self.q)
        return obj

    def __trunc__(self):
        """Truncate."""
        return int(float(self))


def gradient1d(func_or_y, x, args: tuple = None, kwargs: dict = None, eps=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns numerical approximations of 'y' and 'd(y)/dx' given the function 'y=f(x)'.

    Args:
        func_or_y: Either a callable function with the signature y = f(x[, *args][, **kwargs]), or an array of y values
            with a shape of at least (2,).
        x: The value(s) at which the gradient should be computed. If the func_or_y argument is a function, x may take
            any shape. On the other hand, if func_or_y is an array of values the shape of x must match that of y.
        args: A tuple of positional arguments to forward to your function (as per the given signature). Optional.
        kwargs: A dictionary of extra arguments to forward to your function (as per the given signature). Optional.
        eps: Controls the relative stepsize dx when a function is passed to func_or_y. Optional, defaults to 1e-6.

    Returns:
        A tuple of 'y' and 'd(y)/dx'.

    Notes:
        If a function is passed to func_or_y, the values of y returned are only approximated (to save compute). If the
            accuracy of y matters in your use case, compute y beforehand and pass the result in as the first argument.
            By definition, d(y)/dx is numerically approximate.

    """
    if args is not None:
        assert isinstance(args, tuple), f"Expected 'args' to be of type tuple (got {type(args)})"
    if kwargs is not None:
        assert isinstance(kwargs, dict), f"Expected 'kwargs' to be of type dict (got {type(kwargs)})"

    # Recast as necessary
    x = np.atleast_1d(x)

    if callable(func_or_y):

        # STEP 1: Recast x with x-eps/2 and x+eps/2 to obtain xs
        eps = 1e-6 if eps is None else eps
        delta_rel = 1 + eps * np.array([-0.5, 0.5])
        x_broadcasted, delta_rel = broadcast_vector(x, delta_rel)
        x_plusminus_dx = x_broadcasted * delta_rel
        y_plusminus_dy = np.zeros(x_plusminus_dx.shape)

        # STEP 2: Compute y
        if args and kwargs:
            def wrapped_func(xi):
                return func_or_y(xi, *args, **kwargs)
        elif args:
            def wrapped_func(xi):
                return func_or_y(xi, *args)
        elif kwargs:
            def wrapped_func(xi):
                return func_or_y(xi, **kwargs)
        else:
            def wrapped_func(xi):
                return func_or_y(xi)

        # noinspection PyBroadException
        try:
            # Try, check that the passed function supports array arguments
            y_plusminus_dy = wrapped_func(xi=x_plusminus_dx)
        except (Exception) as _:
            # Except any, compute elements one by one
            for i in range(x_plusminus_dx.size):
                y_plusminus_dy.flat[i] = wrapped_func(xi=x_plusminus_dx.flat[i])
        finally:
            # Take an average of the function we ran instead of running function for a 3rd time
            y = np.mean(y_plusminus_dy, axis=0)

        # STEP 3: Numerical differentiation
        dy = np.diff(y_plusminus_dy, axis=0)
        dx = np.diff(x_plusminus_dx, axis=0)
        gradient = (dy / dx).squeeze()  # Squeeze out the dimension we added

    elif args or kwargs:
        error_msg = (
            f"{gradient1d.__name__} was given args/kwargs to pass to 'func_or_y', but "
            f"{type(func_or_y).__name__} object is not callable"
        )
        raise ValueError(error_msg)

    else:
        y = np.atleast_1d(func_or_y)
        assert x.shape[0] >= 2, f"Shape of 'x' needs to be at least two in the first dimension (got dims {x.shape})"
        assert x.shape == y.shape, f"'x' and 'y' do not have the same shape ({x.shape} != {y.shape})"

        # Zero-order estimate of the gradient (completely useless)
        gradient = np.zeros(y.shape)

        # First-order linear correction terms
        dy = np.diff(y)
        dx = np.diff(x)
        m1 = dy / dx  # The slope at points half-way between elements of the x and y arrays (the 'mi' slopes are exact!)

        # Second-order quadratic correction terms
        if dy.shape[0] >= 2:
            d2y = np.diff(m1)
            dx2 = np.diff(x[:-1] + 0.5 * dx)
            m2 = d2y / dx2  # These second order derivatives are located at x[1:-1] (again, this is exact for the array)
        else:
            # The rate of change of dy/dx cannot be determined, assumed zero
            dx2 = m2 = np.zeros((x.shape[0] - 1, *x.shape[1:]))

        # Third-order cubic correction terms
        if dy.shape[0] >= 3:
            d3y = np.diff(m2)
            dx3 = np.diff(x[1:-1])
            m3 = d3y / dx3  # Third order derivatives located in the same place first order ones are (and are exact)
        else:
            # The rate of change of d2y/dx2 cannot be determined, assumed zero
            dx3 = m3 = np.zeros((x.shape[0] - 2, *x.shape[1:]))

        # Interpolate the missing dy/dx gradients
        gradient[:-3] = (
                m1[:-2] - (dx2[:-1] * 1 / 2) * (
                m2[:-1] - (dx3 * 2 / 3) * m3
        ))
        gradient[-3:] = (
                m1[-3:] + (dx2[-3:] * 1 / 2) * (
                m2[-3:] + (dx3[-3:] * 2 / 3) * m3[-3:]
        ))

    return y, gradient
