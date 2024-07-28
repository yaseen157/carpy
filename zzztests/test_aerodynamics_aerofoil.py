import unittest

from matplotlib import pyplot as plt
import numpy as np

from carpy.aerodynamics import PotentialFlow2D
from carpy.utility import GetPath


class PotentialFlowElements(unittest.TestCase):
    """Test elements of potential flow (in 2D)."""

    def test_discretesource(self):
        """Check that the flowfield around a source makes sense."""
        # Define grid
        x = np.linspace(-2, 2, 20)
        z = x.copy()
        X, Z = np.meshgrid(x, z)

        def compute_plotting_params(xgrid, zgrid, sigma):
            """Get direction vectors Uhat and What, and velocity magnitude."""
            U, W = PotentialFlow2D.source_D(
                sigmaj=sigma, x=xgrid, z=zgrid, xj=0, zj=0
            )
            V = (U ** 2 + W ** 2) ** 0.5
            Uhat = U / V  # Normalised U velocities (x-direction)
            What = W / V  # Normalised W velocities (z-direction)
            return Uhat, What, V

        data_sorc = compute_plotting_params(X, Z, sigma=1)
        data_sink = compute_plotting_params(X, Z, sigma=-1)
        data = [data_sorc, data_sink]
        colour = ["blue", "orange"]

        fig, axs = plt.subplots(1, 2, figsize=(6, 3.3), dpi=140)
        fig.suptitle("Discrete Source Element in 2D", size="x-large")

        for i, ax in enumerate(axs.flat):
            ax.quiver(X, Z, *data[i], scale=20, headwidth=6)
            ax.streamplot(X, Z, *data[i][0:2], density=0.5, color=colour[i])
            ax.scatter(0, 0, color="k", s=50, zorder=10)
            ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
            ax.tick_params(
                axis='both',  # changes apply to the both x and y axes
                bottom=False,  # ticks along the bottom edge are off
                left=False,  # ticks along the left edge are off
                labelbottom=False,  # displayed values are off
                labelleft=False
            )
            ax.set_aspect(1)
        else:
            axs[0].set_title("Source")
            axs[1].set_title("Sink")

        figpath = GetPath.localpackage("out/PotentialFlow2D_DiscreteSource.png")
        plt.savefig(figpath)

        self.skipTest(reason=f"Produced file: {figpath}")
        return

    def test_discretedoublet(self):
        """Check that the flowfield around a doublet makes sense."""
        # Define grid
        x = np.linspace(-2, 2, 20)
        z = x.copy()
        X, Z = np.meshgrid(x, z)

        def compute_plotting_params(xgrid, zgrid, mu):
            """Get direction vectors Uhat and What, and velocity magnitude."""
            U, W = PotentialFlow2D.doublet_D(
                muj=mu, x=xgrid, z=zgrid, xj=0, zj=0, beta=None
            )
            V = (U ** 2 + W ** 2) ** 0.5
            Uhat = U / V  # Normalised U velocities (x-direction)
            What = W / V  # Normalised W velocities (z-direction)
            return Uhat, What, V

        data_l = compute_plotting_params(X, Z, mu=1)
        data_r = compute_plotting_params(X, Z, mu=-1)
        data = [data_l, data_r]
        colour = ["blue", "orange"]

        fig, axs = plt.subplots(1, 2, figsize=(6, 3.3), dpi=140)
        fig.suptitle("Discrete Doublet Element in 2D", size="x-large")

        for i, ax in enumerate(axs.flat):
            ax.quiver(X, Z, *data[i], scale=20, headwidth=6)
            ax.streamplot(X, Z, *data[i][0:2], density=0.5, color=colour[i])
            ax.scatter(0, 0, color="k", s=50, zorder=10)
            ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
            ax.tick_params(
                axis='both',  # changes apply to the both x and y axes
                bottom=False,  # ticks along the bottom edge are off
                left=False,  # ticks along the left edge are off
                labelbottom=False,  # displayed values are off
                labelleft=False
            )
            ax.set_aspect(1)

        axs[0].set_title(r"$\mu=+1$")
        axs[1].set_title(r"$\mu=-1$")
        arrow_k = {"width": 0.05, "head_width": 0.2, "color": "k", "zorder": 10}
        axs[0].arrow(0, 0, -1, 0, **arrow_k)
        axs[1].arrow(0, 0, 1, 0, **arrow_k)

        figpath = GetPath.localpackage(
            "out/PotentialFlow2D_DiscreteDoublet.png")
        plt.savefig(figpath)

        self.skipTest(reason=f"Produced file: {figpath}")
        return

    def test_discretevortex(self):
        """Check that the flowfield around a vortex makes sense."""
        # Define grid
        x = np.linspace(-2, 2, 20)
        z = x.copy()
        X, Z = np.meshgrid(x, z)

        def compute_plotting_params(xgrid, zgrid, Gamma):
            """Get direction vectors Uhat and What, and velocity magnitude."""
            U, W = PotentialFlow2D.vortex_D(
                Gammaj=Gamma, x=xgrid, z=zgrid, xj=0, zj=0
            )
            V = (U ** 2 + W ** 2) ** 0.5
            Uhat = U / V  # Normalised U velocities (x-direction)
            What = W / V  # Normalised W velocities (z-direction)
            return Uhat, What, V

        data_cw = compute_plotting_params(X, Z, Gamma=1)
        data_ccw = compute_plotting_params(X, Z, Gamma=-1)
        data = [data_cw, data_ccw]
        colour = ["blue", "orange"]

        fig, axs = plt.subplots(1, 2, figsize=(6, 3.3), dpi=140)
        fig.suptitle("Discrete Vortex Element in 2D", size="x-large")

        for i, ax in enumerate(axs.flat):
            ax.quiver(X, Z, *data[i], scale=20, headwidth=6)
            ax.streamplot(X, Z, *data[i][0:2], density=0.5, color=colour[i])
            ax.scatter(0, 0, color="k", s=50, zorder=10)
            ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
            ax.tick_params(
                axis='both',  # changes apply to the both x and y axes
                bottom=False,  # ticks along the bottom edge are off
                left=False,  # ticks along the left edge are off
                labelbottom=False,  # displayed values are off
                labelleft=False
            )
            ax.set_aspect(1)
        else:
            axs[0].set_title("Clockwise")
            axs[1].set_title("Anti-clockwise")

        figpath = GetPath.localpackage("out/PotentialFlow2D_DiscreteVortex.png")
        plt.savefig(figpath)

        self.skipTest(reason=f"Produced file: {figpath}")
        return

    def test_constantsource(self):
        """Check that the flowfield around a source makes sense."""
        # Define grid
        x = np.linspace(-2, 2, 20)
        z = x.copy()
        X, Z = np.meshgrid(x, z)

        # Define source panel
        panel = {"xj0": 1.5, "xj1": -1.5, "zj0": 0, "zj1": 0}

        def compute_plotting_params(xgrid, zgrid, sigma):
            """Get direction vectors Uhat and What, and velocity magnitude."""
            U, W = PotentialFlow2D.source_C(
                sigmaj=sigma, x=xgrid, z=zgrid, **panel
            )
            V = (U ** 2 + W ** 2) ** 0.5
            Uhat = U / V  # Normalised U velocities (x-direction)
            What = W / V  # Normalised W velocities (z-direction)
            return Uhat, What, V

        data_sorc = compute_plotting_params(X, Z, sigma=1)
        data_sink = compute_plotting_params(X, Z, sigma=-1)
        data = [data_sorc, data_sink]
        colour = ["blue", "orange"]

        fig, axs = plt.subplots(1, 2, figsize=(6, 3.3), dpi=140)
        fig.suptitle("Constant Strength Source Panel in 2D", size="x-large")

        for i, ax in enumerate(axs.flat):
            ax.quiver(X, Z, *data[i], scale=20, headwidth=6)
            ax.streamplot(X, Z, *data[i][0:2], density=0.5, color=colour[i])
            ax.plot(
                [panel["xj0"], panel["xj1"]],
                [panel["zj0"], panel["zj1"]],
                c="k", lw=3
            )
            ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
            ax.tick_params(
                axis='both',  # changes apply to the both x and y axes
                bottom=False,  # ticks along the bottom edge are off
                left=False,  # ticks along the left edge are off
                labelbottom=False,  # displayed values are off
                labelleft=False
            )
            ax.set_aspect(1)
        else:
            axs[0].set_title("Source")
            axs[1].set_title("Sink")

        figpath = GetPath.localpackage("out/PotentialFlow2D_ConstantSource.png")
        plt.savefig(figpath)

        self.skipTest(reason=f"Produced file: {figpath}")
        return

    def test_constantdoublet(self):
        """Check that the flowfield around a doublet makes sense."""
        # Define grid
        x = np.linspace(-2, 2, 20)
        z = x.copy()
        X, Z = np.meshgrid(x, z)

        # Define source panel
        panel = {"xj0": 1.5, "xj1": -1.5, "zj0": 0, "zj1": 0}

        def compute_plotting_params(xgrid, zgrid, mu):
            """Get direction vectors Uhat and What, and velocity magnitude."""
            U, W = PotentialFlow2D.doublet_C(
                muj=mu, x=xgrid, z=zgrid, **panel
            )
            V = (U ** 2 + W ** 2) ** 0.5
            Uhat = U / V  # Normalised U velocities (x-direction)
            What = W / V  # Normalised W velocities (z-direction)
            return Uhat, What, V

        data_sorc = compute_plotting_params(X, Z, mu=1)
        data_sink = compute_plotting_params(X, Z, mu=-1)
        data = [data_sorc, data_sink]
        colour = ["blue", "orange"]

        fig, axs = plt.subplots(1, 2, figsize=(6, 3.3), dpi=140)
        fig.suptitle("Constant Strength Doublet Panel in 2D", size="x-large")

        for i, ax in enumerate(axs.flat):
            ax.quiver(X, Z, *data[i], scale=20, headwidth=6)
            ax.streamplot(X, Z, *data[i][0:2], density=0.5, color=colour[i])
            ax.plot(
                [panel["xj0"], panel["xj1"]],
                [panel["zj0"], panel["zj1"]],
                c="k", lw=3
            )
            ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
            ax.tick_params(
                axis='both',  # changes apply to the both x and y axes
                bottom=False,  # ticks along the bottom edge are off
                left=False,  # ticks along the left edge are off
                labelbottom=False,  # displayed values are off
                labelleft=False
            )
            ax.set_aspect(1)
        else:
            # axs[0].set_title("<=\n<=\n<=", rotation=90, size="medium")
            # axs[0].set_xlabel("<=\n<=\n<=", rotation=90, size="medium")
            # axs[1].set_title("=>\n=>\n=>", rotation=90, size="medium")
            # axs[1].set_xlabel("=>\n=>\n=>", rotation=90, size="medium")
            axs[0].set_title("Downwash")
            axs[1].set_title("Upwash")

        figpath = GetPath.localpackage(
            "out/PotentialFlow2D_ConstantDoublet.png")
        plt.savefig(figpath)

        self.skipTest(reason=f"Produced file: {figpath}")
        return

    def test_constantvortex(self):
        """Check that the flowfield around a vortex makes sense."""
        # Define grid
        x = np.linspace(-2, 2, 20)
        z = x.copy()
        X, Z = np.meshgrid(x, z)

        # Define source panel
        panel = {"xj0": 1.5, "xj1": -1.5, "zj0": 0, "zj1": 0}

        def compute_plotting_params(xgrid, zgrid, gamma):
            """Get direction vectors Uhat and What, and velocity magnitude."""
            U, W = PotentialFlow2D.vortex_C(
                gammaj=gamma, x=xgrid, z=zgrid, **panel
            )
            V = (U ** 2 + W ** 2) ** 0.5
            Uhat = U / V  # Normalised U velocities (x-direction)
            What = W / V  # Normalised W velocities (z-direction)
            return Uhat, What, V

        data_cw = compute_plotting_params(X, Z, gamma=1)
        data_ccw = compute_plotting_params(X, Z, gamma=-1)
        data = [data_cw, data_ccw]
        colour = ["blue", "orange"]

        fig, axs = plt.subplots(1, 2, figsize=(6, 3.3), dpi=140)
        fig.suptitle("Constant Strength Vortex Panel in 2D", size="x-large")

        for i, ax in enumerate(axs.flat):
            ax.quiver(X, Z, *data[i], scale=20, headwidth=6)
            ax.streamplot(X, Z, *data[i][0:2], density=0.5, color=colour[i])
            ax.plot(
                [panel["xj0"], panel["xj1"]],
                [panel["zj0"], panel["zj1"]],
                c="k", lw=3
            )
            ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
            ax.tick_params(
                axis='both',  # changes apply to the both x and y axes
                bottom=False,  # ticks along the bottom edge are off
                left=False,  # ticks along the left edge are off
                labelbottom=False,  # displayed values are off
                labelleft=False
            )
            ax.set_aspect(1)
        else:
            axs[0].set_title("Clockwise")
            axs[1].set_title("Anti-clockwise")

        figpath = GetPath.localpackage("out/PotentialFlow2D_ConstantVortex.png")
        plt.savefig(figpath)

        self.skipTest(reason=f"Produced file: {figpath}")
        return


if __name__ == '__main__':
    unittest.main()
