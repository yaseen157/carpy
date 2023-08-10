"""Simple Vortex Lattice Method for thin aerofoils."""
import numpy as np

from carpy.utility import Hint, cast2numpy

__all__ = []
__author__ = "Yaseen Reza"


def vfil(xyzA: Hint.nums, xyzB: Hint.nums, xyzC: Hint.nums):
    """
    Compute the influence of vortex line AB at control point C, return zero for
    small denominators (to exclude self-influence).

    Args:
        xyzA: Vector describing position of control point A in 3D space.
        xyzB: Vector describing position of control point B in 3D space.
        xyzC: Vector describing position of control point C in 3D space.

    Returns:
        Influence I of vortex line AB at control point C.

    """
    # Recast as necessary
    xyzA, xyzB, xyzC = cast2numpy([xyzA, xyzB, xyzC])

    small = 1e-12
    ab = xyzB - xyzA
    ac = xyzC - xyzA
    bc = xyzC - xyzB
    cross = np.cross(ac, bc)
    den = 4 * np.pi * (cross ** 2).sum()
    abs_ac = np.sqrt((ac ** 2).sum())
    abs_bc = np.sqrt((bc ** 2).sum())
    num = (ab * (ac / abs_ac - bc / abs_bc)).sum()

    if den <= small:
        I = np.zeros(3)
    else:
        I = num / den * cross

    return I


def rectwing(alpha, AR, N):
    """

    Args:
        alpha ():
        AR ():
        N ():

    Returns:
        tuple: (lift coefficient, (induced) drag coefficient)

    """
    large = 1e6
    Q = [np.cos(alpha), 0.0, np.sin(alpha)]  # Freestream orientation
    bw = AR  # Wing, span
    cw = 1.0  # Wing, chord
    factor = np.sqrt(N / (N + 1))
    bp = bw * factor  # Wing, control span
    cp = np.array([0.5 * cw / factor, 0, 0])  # Wing, control chord

    # Locate vortex segments, control, and normal vectors
    dy = np.array([0, bp / N, 0])  # Horseshoe discretise
    xa, xb, xc = np.zeros((3, N, 3))  # Store vortex vertices (a,b) and ctrl (c)
    n = np.zeros((N, 3))  # Create empty normal vector
    xa[0] = np.array([0, -0.5 * bp, 0])
    xb[0] = np.array([0, -0.5 * bp, 0]) + dy
    for i in range(1, N):
        xa[i] = xb[i - 1]
        xb[i] = xa[i] + dy
    for i in range(N):
        xc[i] = 0.5 * (xa[i] + xb[i]) + cp
        n[i] = np.array([0, 0, 1])  # z-direction is the chord line normal

    # Set matrix and right hand side elements, solve for circulations (Gamma)
    A = np.zeros((N, N))
    rhs = np.zeros((N, 1))

    for i in range(N):
        for j in range(N):
            # Influence contributions from 'LARGE -> xa -> xb -> LARGE' on xc
            I = vfil(xa[j], xb[j], xc[i])
            I += vfil(xa[j] + np.array([large, 0, 0]), xa[j], xc[i])
            I += vfil(xb[j], xb[j] + np.array([large, 0, 0]), xc[i])
            A[i, j] = (I * n).sum()
        rhs[i] = -(Q * n).sum()

    # Sum of circulation in z-direction should match freestream, given a gamma
    gamma = np.linalg.solve(A, rhs)

    # Forces at centres of bound vortices (parallel with leading edge)
    Fx, Fy, Fz = np.zeros((3, N))
    bc = 0.5 * (xa + xb)
    for i in range(N):
        # Local velocity vector on each load element i
        u = Q
        for j in range(N):
            u += vfil(xa[j], xb[j], bc[i]) * gamma[j]
            u += vfil(xa[j] + np.array([large, 0, 0]), xa[j], bc[i]) * gamma[j]
            u += vfil(xb[j], xb[j] + np.array([large, 0, 0]), bc[i]) * gamma[j]
        # u cross s gives direction of action of the force from circulation
        s = xb[i] - xa[i]
        Fx[i], Fy[i], Fz[i] = np.cross(u, s) * gamma[i]

    # Resolve these forces into perpendicular and parallel to freestream
    Cl = (np.sum(Fz) * np.cos(alpha) - np.sum(Fx) * np.sin(alpha)
          ) / (0.5 * bw * cw)
    Cdi = (np.sum(Fx) * np.cos(alpha) + np.sum(Fz) * np.sin(alpha)
           ) / (0.5 * bw * cw)

    return Cl, Cdi


if __name__ == "__main__":
    ARs = np.arange(1e-2, 21, 1)
    deltas = []
    for AR in ARs:
        Cl, Cdi = rectwing(np.radians(1), AR, N=40)
        delta = ((Cdi * np.pi * AR) / (Cl ** 2)) - 1
        deltas.append(delta)
    from matplotlib import pyplot as plt

    plt.plot(ARs, deltas)
    plt.show()
