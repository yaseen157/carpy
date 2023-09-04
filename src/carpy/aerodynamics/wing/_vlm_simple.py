"""Simple Vortex Lattice Method for thin aerofoils."""
import numpy as np
from scipy.integrate import trapezoid

from carpy.utility import Hint, cast2numpy

__all__ = []
__author__ = "Yaseen Reza"


def vfil(xyzA: Hint.nums, xyzB: Hint.nums, xyzC: Hint.nums):
    """
    Compute the influence of vortex line AB at control point C, return zero for
    small denominators (to exclude self-influence).

    Args:
        xyzA: Vector describing position of reference point A in 3D space.
        xyzB: Vector describing position of reference point B in 3D space.
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
        alpha: Freestream angle of attack.
        AR: Rectangular wing aspect ratio.
        N: Number of horseshoe vortices.

    Returns:
        tuple: (lift coefficient, (induced) drag coefficient)

    """
    large = 1e6
    Q = [np.cos(alpha), 0.0, np.sin(alpha)]  # Freestream orientation
    bw = AR  # Wingspan (== AR when chord length of the wing is 1)
    cw = 1.0  # Wingchord (== 1, constant reference length)
    factor = np.sqrt(N / (N + 1))  # For 1/sqrt(2) when N == 1
    bp = bw * factor  # Wingspan, prime
    cp = np.array([0.5 * cw / factor, 0, 0])  # Wingchord, prime

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
    wingarea = bw * cw
    Cl = np.sum(Fz * np.cos(alpha) - Fx * np.sin(alpha)) / (wingarea / 2)
    Cdi = np.sum(Fx * np.cos(alpha) + Fz * np.sin(alpha)) / (wingarea / 2)

    print(f"Fx={np.sum(Fx)}")
    print(f"Fz={np.sum(Fz)}")
    print(f"{wingarea=:.2f}, {Cl=:.2f}, {Cdi=:.2f}")

    return Cl, Cdi


class VLMSolutionRigid(object):

    def __init__(self, sections, span: Hint.num, alpha: Hint.num = None,
                 mirror: bool = None, N: int = None):
        """
        Args:
            sections: Wing sections object, defining 3D geometry.
            span: Tip-to-tip span of wing (if mirrored), else root-to-tip span.
            alpha: Angle of attack at the wing root (station zero). Optional,
                defaults to zero angle of attack.
            mirror: Whether or not to treat the wing sections object as
                symmetrical around the smallest station index. Optional,
                defaults to True (mirrored wing).
            N: The number of horseshoe vortices. Optional, defaults to 40.

        Notes:
            The reference coordinate system used in this code adheres to the
            NorthEastDown (NED) standard aircraft principle axes representation.

        """
        # Recast as necessary
        alpha = 0 if alpha is None else float(alpha)
        mirror = True if mirror is None else mirror
        N = 40 if N is None else int(N)

        # Problem setup
        large = 1e6
        Q = np.array([-np.cos(alpha), 0.0, -np.sin(alpha)])  # Freestream orient
        scalefactor = np.sqrt(N / (N + 1))  # Scaling fix for N = 1 case
        # Linearly spaced sections in defined span
        if mirror is True:
            if N % 2 == 0:  # Even number of horseshoes, don't define @y=0
                Nsections = sections[::N][1::2]
                Nsections = Nsections[::-1] + Nsections
            else:  # Odd number of horseshoes, definition @y=0 is required
                Nsections = sections[::int(np.ceil(N / 2))]  # ceil includes y=0
                Nsections = Nsections[:0:-1] + Nsections
        else:
            Nsections = sections[::N]
        # bprime = combined span of all horseshoe elements
        bprime = span * scalefactor
        # cprime = longitudinal dist. between control point and horseshoe front
        chord = np.array([section.chord for section in Nsections])
        cprime = np.array([[-0.5 * cw / scalefactor, 0, 0] for cw in chord])

        # Locate vortex segments (a,b), control vectors (c), and normal vectors
        va, vb, vc, n = np.zeros((4, N, 3))  # Initial assignment to 3D zeros.
        # ... distribute vortex locators (va, vb) in the y-direction
        if mirror is True:
            ys = np.linspace(-bprime / 2, bprime / 2, N + 1)
        else:
            ys = np.linspace(0, bprime, N + 1)
        va[:, 1] = ys[:-1]
        vb[:, 1] = ys[1:]
        # ... prescribe normal vectors as if wing is a flat plate
        n[:, 2] = -1.0

        # Define horseshoe geometry in the y >= 0, +ve direction
        dy = bprime / N
        for i in range(N):

            # Port-side vortices (any mirrored elements, if they exist)
            if va[i, 1] < 0 and vb[i, 1] <= 0:
                continue  # skip

            # Centreline vortex (vortex symmetrical about the centreline)
            elif va[i, 1] < 0:
                # Create rotation matrix for aerodynamic + geometric twist
                # section.alpha_zl automatically combines aero + geo twist...
                cos_theta = np.cos(-Nsections[i].alpha_zl)
                sin_theta = np.sin(-Nsections[i].alpha_zl)
                rot_twist = np.array([
                    [cos_theta, 0, sin_theta],
                    [0, 1, 0],  # About the Y-axis
                    [-sin_theta, 0, cos_theta]
                ])
                del cos_theta, sin_theta

                # Apply rotation to normal vector and control point locator
                n[i] = rot_twist @ n[i]
                vc[i] = (va[i] + vb[i]) / 2 + (rot_twist @ cprime[i])

            # Starboard vortices
            else:
                # Sweep vortex locators (Move x-coordinates)
                tan_sweep = np.tan(Nsections[i].sweep)
                va[i + 1:, 0] -= dy * tan_sweep  # Outboard sections
                vb[i + 1:, 0] -= dy * tan_sweep
                va[i, 0] -= (dy / 2) * tan_sweep  # Current section
                vb[i, 0] -= (dy / 2) * tan_sweep
                del tan_sweep

                # Apply dihedral to vortex locators (Move z-coordinates)
                tan_dihedral = np.tan(Nsections[i].dihedral)
                va[i + 1:, 2] -= dy * tan_dihedral  # Outboard sections
                vb[i + 1:, 2] -= dy * tan_dihedral
                va[i, 2] -= (dy / 2) * tan_dihedral  # Current section
                vb[i, 2] -= (dy / 2) * tan_dihedral
                del tan_dihedral

                # Create rotation matrix for aerodynamic + geometric twist
                # section.alpha_zl automatically combines aero + geo twist...
                cos_theta = np.cos(-Nsections[i].alpha_zl)
                sin_theta = np.sin(-Nsections[i].alpha_zl)
                rot_twist = np.array([
                    [cos_theta, 0, sin_theta],
                    [0, 1, 0],  # About the Y-axis
                    [-sin_theta, 0, cos_theta]
                ])
                del cos_theta, sin_theta

                # Create rotation matrix for dihedral
                # Take negatives, because matrix rotation assumes +ve==clockwise
                cos_gamma = np.cos(-Nsections[i].dihedral)
                sin_gamma = np.sin(-Nsections[i].dihedral)
                rot_dihedral = np.array([
                    [1, 0, 0],  # About the X-axis
                    [0, cos_gamma, -sin_gamma],
                    [0, sin_gamma, cos_gamma]
                ])
                del cos_gamma, sin_gamma

                # Apply rotations to normal vector and control point locator
                n[i] = rot_dihedral @ (rot_twist @ n[i])
                vc[i] = (va[i] + vb[i]) / 2 + (rot_twist @ cprime[i])

        # Now if necessary (wing mirrored), copy starboard geometry to port side
        else:
            if mirror is True:
                mirror_idx = int(N / 2)
                # For consistency, va and vb must swap in the mirror image
                va[:mirror_idx] = vb[-mirror_idx:][::-1] * np.array([1, -1, 1])
                vb[:mirror_idx] = va[-mirror_idx:][::-1] * np.array([1, -1, 1])
                # ", but mirroring vc
                vc[:mirror_idx] = vc[-mirror_idx:][::-1] * np.array([1, -1, 1])
                # Simple mirror for normal vectors
                n[:mirror_idx] = n[-mirror_idx:][::-1]

        # Set matrix and right hand side elements, solve for circulations (Gamma)
        A = np.zeros((N, N))
        rhs = np.zeros((N, 1))

        for i in range(N):
            for j in range(N):
                # Influence contributions; 'LARGE -> va -> vb -> LARGE' on vc
                I = vfil(va[j], vb[j], vc[i])
                I += vfil(va[j] + np.array([-large, 0, 0]), va[j], vc[i])
                I += vfil(vb[j], vb[j] + np.array([-large, 0, 0]), vc[i])
                A[i, j] = (I * n).sum()
            rhs[i] = -(Q * n).sum()

        # Sum of circulation in z-direction should match freestream, given a gamma
        gamma = np.linalg.solve(A, rhs)

        # Forces at centres of bound vortices (parallel with leading edge)
        Fx, Fy, Fz = np.zeros((3, N))
        bc = 0.5 * (va + vb)
        for i in range(N):
            # Local velocity vector on each load element i
            u = np.copy(Q)
            for j in range(N):
                u += vfil(va[j], vb[j], bc[i]) * gamma[j]
                u += vfil(va[j] + np.array([-large, 0, 0]), va[j], bc[i]) * \
                     gamma[j]
                u += vfil(vb[j], vb[j] + np.array([-large, 0, 0]), bc[i]) * \
                     gamma[j]
            # u cross s gives direction of action of the force from circulation
            s = vb[i] - va[i]
            Fx[i], Fy[i], Fz[i] = np.cross(u, s) * gamma[i]

        # Resolve these forces into perpendicular and parallel to freestream
        wingarea = trapezoid(sections[::100].chord, dx=span / 100)
        Cl = np.sum(Fx * np.sin(alpha) - Fz * np.cos(alpha)) / (wingarea / 2)
        Cdi = np.sum(-Fx * np.cos(alpha) - Fz * np.sin(alpha)) / (wingarea / 2)

        print(f"Fx={np.sum(Fx)}")
        print(f"Fz={np.sum(Fz)}")
        print(f"{wingarea=:.2f}, {Cl=:.2f}, {Cdi=:.2f}")

        return


if __name__ == "__main__":
    from carpy.aerodynamics.aerofoil import NewNDAerofoil
    from carpy.aerodynamics.wing import WingSections, WingSection

    n0012 = NewNDAerofoil.from_procedure.NACA("0012")
    n2412 = NewNDAerofoil.from_procedure.NACA("2412")

    # Define buttock-line geometry
    mysections = WingSections()
    mysections[0] = WingSection(n0012)
    mysections[60] = mysections[0].deepcopy()
    mysections[100] = WingSection(n0012)

    # Add sweep and dihedral
    # mysections[:60].sweep = np.radians(0)
    # mysections[60:].sweep = np.radians(2)
    # mysections[0:].dihedral = np.radians(3)
    mysections[:].sweep = 0
    mysections[:].dihedral = 0

    # Introduce wing taper
    mysections[0:].chord = 1.0
    # mysections[100].chord = 0.4

    VLMSolutionRigid(sections=mysections, span=30, alpha=np.radians(8), N=5)
    rectwing(alpha=np.radians(8), AR=30, N=5)
