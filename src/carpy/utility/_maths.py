"""A module of maths utilities."""
import numpy as np

from carpy.utility._miscellaneous import Hint, cast2numpy

__all__ = ["interp_lin", "interp_exp", "moving_average", "point_diff",
           "point_curvature"]
__author__ = "Yaseen Reza"


def interp_lin(x: Hint.nums, xp: Hint.nums, fp: Hint.nums,
               bounded: bool = True) -> Hint.nums:
    """
    A linear interpolation function offering unbounded extrapolation.

    Args:
        x: The values which should be fed to the interpolation function.
        xp: Sorted list of x-values to draw interpolated input from.
        fp: List of return values to draw interpolated outputs from.
            bounded: Boolean, selects whether output is bounded by 'fp' values.
        bounded: If true, limit output by fp limits. Otherwise, extrapolate.

    Returns:
        np.ndarray: The linearly-interpolated values.

    """
    # Recast as necessary
    x = cast2numpy(x)
    xp = cast2numpy(xp)
    fp = cast2numpy(fp)

    # Prepare differentials between each point
    dfp = np.diff(fp)
    dxp = np.diff(xp)

    out = np.zeros_like(fp, shape=x.shape)  # Shape of x, object style of fp

    for i in range(len(dfp)):
        myslice = (xp[i] <= x) & (x <= xp[i + 1])
        out[myslice] = (x[myslice] - xp[i]) * (dfp[i] / dxp[i]) + fp[i]

    sliceL, sliceR = x < xp[0], x > xp[-1]
    if bounded is True:
        out[sliceL] = fp[0]
        out[sliceR] = fp[-1]
    else:
        out[sliceL] = (x[sliceL] - xp[0]) * (dfp / dxp)[0] + fp[0]
        out[sliceR] = (x[sliceR] - xp[-1]) * (dfp / dxp)[-1] + fp[-1]

    return out


def interp_exp(x: Hint.nums, xp: Hint.nums, fp: Hint.nums,
               bounded: bool = True) -> Hint.nums:
    """
    A linear interpolation function offering unbounded extrapolation.

    Args:
        x: The values which should be fed to the interpolation function.
        xp: Sorted list of x-values to draw interpolated input from.
        fp: List of return values to draw interpolated outputs from.
            bounded: Boolean, selects whether output is bounded by 'fp' values.
        bounded: If true, limit output by fp limits. Otherwise, extrapolate.

    Returns:
        np.ndarray: The exponentially-interpolated values.

    """
    # Recast as necessary
    x = cast2numpy(x)
    xp = cast2numpy(xp)
    fp = cast2numpy(fp)

    # Prepare differentials between each point
    log_dfp = np.log(fp[1:] / fp[:-1])
    lin_dxp = xp[1:] - xp[:-1]
    b = log_dfp / lin_dxp
    a = fp[:-1] / np.exp(b * xp[:-1])

    out = np.zeros_like(fp, shape=x.shape)  # Shape of x, object style of fp

    for i in range(len(log_dfp)):
        myslice = (xp[i] <= x) & (x <= xp[i + 1])
        out[myslice] = a[i] * np.exp(b[i] * x[myslice])

    sliceL, sliceR = x < xp[0], x > xp[-1]
    if bounded is True:
        out[sliceL] = fp[0]
        out[sliceR] = fp[-1]
    else:
        out[sliceL] = a[0] * np.exp(b[0] * x[sliceL])
        out[sliceR] = a[-1] * np.exp(b[-1] * x[sliceR])

    return out


def moving_average(x: Hint.nums, w: int = None) -> np.ndarray:
    """
    Compute the moving average of an array.

    Args:
        x: 1D array which should have the moving average applied.
        w: The size of the moving average window. Optional, defaults to 2.

    Returns:
        An array of size n-(w-1) (when given an input array x of size n) and a
            moving average window size of w.

    """
    # Recast as necessary
    x = cast2numpy(x)
    w = 2 if w is None else int(2)

    # https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    return np.convolve(x, np.ones(w), 'valid') / w


def point_diff(y: Hint.num, x: Hint.num = None) -> np.ndarray:
    """
    Given coordinate arrays describing discrete points, find the gradient at
    (and not between!) the points.

    Args:
        y: Y-values of the discrete points.
        x: X-values of the discrete points. Optional, assumes the form of
            range(len(y)) if not specified.

    Returns:
        The averaged differentials of the array, mapped back to the same shape
            and same x-ordinates of the original input arrays.

    """
    # Recast as necessary
    y = cast2numpy(y)
    x = np.arange(len(y)) if x is None else cast2numpy(x)

    assert x.shape == y.shape, "Expected homogeneity of array shapes"
    assert x.ndim == y.ndim == 1, f"Unsupported array dimensions, check for 1D"

    # Differentiation to give the gradients between points described in arrays
    dydx_mid = np.diff(y) / np.diff(x)

    # Gradient at the points is described by averaging the prior result
    dydx_pts = np.zeros(x.shape)
    # dydx_pts[0], dydx_pts[-1] = dydx_mid[0], dydx_mid[-1]  # copy ends
    dydx_pts[1:-1] = moving_average(dydx_mid, 2)
    dydx_pts[0], dydx_pts[-1] = dydx_pts[1], dydx_pts[-2]  # copy ends

    return dydx_pts


def point_curvature(x: Hint.num, y: Hint.num) -> np.ndarray:
    """
    Given arrays describing points in 2D, determine the signed curvature at each
    point.

    Args:
        x: 1D array of coordinates, the abscissa; x-coordinate.
        y: 1D array of coordinates, the ordinates; y-coordinate.

    Returns:
        An array describing the signed curvature at each point.

    """
    # Recast as array
    x = cast2numpy(x)
    y = cast2numpy(y)

    # Requisite differentations
    xprime = point_diff(x)
    xpprime = point_diff(xprime)
    yprime = point_diff(y)
    ypprime = point_diff(yprime)

    # Compute
    numerator = xprime * ypprime - yprime * xpprime
    denominator = (xprime ** 2 + yprime ** 2) ** (3 / 2)
    signed_curvature = numerator / denominator

    return signed_curvature
