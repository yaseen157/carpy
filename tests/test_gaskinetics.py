import unittest

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from carpy.gaskinetics import (
    V_max, nu_max, IsentropicFlow, NormalShock, ObliqueShock,
    ExpansionFan, RayleighFlow, FannoFlow
)
from carpy.utility import GetPath


class SimpleWaveFlow(unittest.TestCase):
    """Reproduce figures from ESDU."""

    def test_S00_03_08_fig1a(self):
        """Figure 1a) Mach angle versus Mach number."""

        fig, ax = plt.subplots(1, dpi=140, figsize=(6, 5))

        # Create 3 lines
        for Mstart in [1, 2, 3]:
            Ms = np.arange(Mstart, Mstart + 1, 1e-2)
            mus = np.degrees(IsentropicFlow.mu(Ms))
            ax.plot(Ms % 1, mus, c="k")

            # Label Mach
            text_xs = np.arange(0.1, 1.1, 0.1)
            text_ys = np.degrees(IsentropicFlow.mu(text_xs + Mstart)) + 1
            for i in range(10):
                string = f"{text_xs[i] + Mstart:.1f}"
                ax.text(text_xs[i], text_ys[i], string, fontsize=7)
            else:
                Mx = 0.55
                My = np.degrees(IsentropicFlow.mu(Mx + Mstart)) + 3
                ax.text(Mx, My, "M", fontsize=7)

        # Make plot pretty
        ax.set_xlim(0, 1.1)
        ax.set_ylim(10, 90)
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False
        )
        ax.set_ylabel("$\mu$ [degrees]")
        ax.spines[["top", "right", "bottom"]].set_visible(False)  # Hide borders

        # Grid control
        major_ticks = np.arange(0, 1.1, 0.1)
        minor_ticks = np.arange(0, 1.1, 0.01)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        major_ticks = np.arange(10, 91, 10)
        minor_ticks = np.arange(10, 91, 1)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_aspect(0.1 / 10)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        figpath = GetPath.localpackage("out/ESDU S.00.03.08 Fig1a.png")
        plt.savefig(figpath)

        self.skipTest(reason=f"Produced file: {figpath}")
        return

    def test_S00_03_08_fig1b(self):
        """Figure 1b) Flow deflection angle carpet plot."""

        fig, ax = plt.subplots(1, dpi=140, figsize=(6, 5))

        M2s_contour = np.arange(1.0, 4.25, 0.25)
        gammas_contour = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 5 / 3])

        gammas = np.linspace(gammas_contour.min(), gammas_contour.max())
        omegas = np.linspace(0, 100)  # In degrees

        mesh_gamma, mesh_omega = np.meshgrid(gammas, omegas)
        mesh_M2s = np.zeros(mesh_gamma.shape)
        xs = np.ones(len(mesh_omega))[:, None] * np.arange(mesh_omega.shape[1])

        for j, gamma in enumerate(gammas):
            for i, omega in enumerate(omegas):
                mesh_M2s[j, i] = \
                    ExpansionFan(M1=1, gamma=gamma, theta=np.radians(omega)).M2

        # griddata(
        #     points=zip(xs.flat, mesh_omega.flat),
        #     values=mesh_gamma.flatten(),
        #     xi=zip(xs.flat, )
        # )
        # ax.contour(mesh_gamma, mesh_omega, mesh_M2s, levels=M2s_contour)
        # ax.set_ylim(0, 100)
        # ax.contour(xs, mesh_omega, mesh_M2s, levels=M2s_contour)
        ax.contour(xs, mesh_omega, mesh_gamma, levels=gammas_contour)
        #
        # plt.show()

        self.skipTest(reason="This test is not ready yet.")

        return
