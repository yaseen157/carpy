"""A module of maths utilities."""
import numpy as np

__all__ = ["gradient1d"]
__author__ = "Yaseen Reza"


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
        x_broadcasted = np.broadcast_to(x, (*delta_rel.shape, *x.shape))
        delta_rel = np.expand_dims(delta_rel, tuple(range(x_broadcasted.ndim - 1))).T
        x_plusminus_dx = x * delta_rel
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
        except Exception as _:
            # Except any, compute elements one by one
            for i in range(x_plusminus_dx.size):
                y_plusminus_dy.flat[i] = wrapped_func(xi=x.flat[i])
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
