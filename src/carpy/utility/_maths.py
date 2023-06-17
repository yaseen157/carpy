"""A module of maths utilities."""
import numpy as np

from carpy.utility._miscellaneous import Hint, cast2numpy

__all__ = ["interp_lin", "interp_exp"]
__author__ = "Yaseen Reza"


def interp_lin(x: Hint.nums, xp: Hint.nums, fp: Hint.nums,
               bounded: bool = True) -> np.ndarray:
    """
    A linear interpolation function offering unbounded extrapolation.

    Args:
        x: The values which should be fed to the interpolation function.
        xp: Sorted list of x-values to draw interpolated input from.
        fp: List of return values to draw interpolated outputs from.
            bounded: Boolean, selects whether output is bounded by 'fp' values.
        bounded: If true, output is limited by fp limits. Otherwise extrapolate.

    Returns:
        np.ndarray: The linearly-interpolated values.

    """
    # Recast as necessary
    x = cast2numpy(x)
    xp = cast2numpy(xp)
    fp = cast2numpy(fp)

    # Prepare differentials between each point
    dfp = fp[1:] - fp[:-1]
    dxp = xp[1:] - xp[:-1]

    out = np.zeros_like(x)

    for i, (dfpi, dxpi) in enumerate(zip(dfp, dxp)):
        out = np.where(
            (xp[i] <= x) & (x <= xp[i + 1]),
            (x - xp[i]) * (dfpi / dxpi) + fp[i],
            out
        )
    else:
        del i, dfpi, dxpi  # Clear the namespace a little

    if bounded is True:
        out = np.where(x < xp[0], fp[0], out)
        out = np.where(x > xp[-1], fp[-1], out)
    else:
        out = np.where(x < xp[0], (x - xp[0]) * (dfp / dxp)[0] + fp[0], out)
        out = np.where(xp[-1] < x, (x - xp[-1]) * (dfp / dxp)[-1] + fp[-1], out)

    return out


def interp_exp(x: Hint.nums, xp: Hint.nums, fp: Hint.nums,
               bounded: bool = True) -> np.ndarray:
    """
    A linear interpolation function offering unbounded extrapolation.

    Args:
        x: The values which should be fed to the interpolation function.
        xp: Sorted list of x-values to draw interpolated input from.
        fp: List of return values to draw interpolated outputs from.
            bounded: Boolean, selects whether output is bounded by 'fp' values.
        bounded: If true, output is limited by fp limits. Otherwise extrapolate.

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

    out = np.zeros_like(x)

    for i, (ai, bi) in enumerate(zip(a, b)):
        out = np.where(
            (xp[i] <= x) & (x <= xp[i + 1]),
            ai * np.exp(bi * x),
            out
        )
    else:
        del i, ai, bi  # Clear the namespace a little

    if bounded is True:
        out = np.where(x < xp[0], fp[0], out)
        out = np.where(x > xp[-1], fp[-1], out)
    else:
        out = np.where(x < xp[0], a[0] * np.exp(b[0] * x), out)
        out = np.where(x > xp[-1], a[-1] * np.exp(b[-1] * x), out)

    return out
