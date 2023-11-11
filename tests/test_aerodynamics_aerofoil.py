import unittest

from matplotlib import pyplot as plt
import numpy as np

from carpy.aerodynamics.aerofoil import NewAerofoil, ThinAerofoil
from carpy.aerodynamics.aerofoil._solutions_lsaero import VOR2D, SORC2D
from carpy.utility import GetPath


class Profiles(unittest.TestCase):
    """Methods to test aerofoil section generation and file parsing."""

    def test_generateNACA(self):
        # Four digit series
        n0012 = NewAerofoil.from_method.NACA("0012")
        n2412 = NewAerofoil.from_method.NACA("2412")
        n2412_63 = NewAerofoil.from_method.NACA("2412-63")
        # Five digit series
        n23012 = NewAerofoil.from_method.NACA("23012")
        n23012_45 = NewAerofoil.from_method.NACA("23012-45")
        n44112 = NewAerofoil.from_method.NACA("44112")
        # 16-series
        n16_012 = NewAerofoil.from_method.NACA("16-012")
        n16_912_3 = NewAerofoil.from_method.NACA("16-912,a=0.3")
        return


class ThinAerofoilTheory(unittest.TestCase):
    """Methods to test thin aerofoil theory."""

    def test_liftcurveslope(self):
        """Thin aerofoil theory suggests ideal lift slope of 2 pi."""
        flatplate = NewAerofoil.from_method.NACA("0001")
        solution = ThinAerofoil(aerofoil=flatplate, alpha=0)
        Clalpha = solution.CLalpha
        self.assertAlmostEqual(Clalpha, 2 * 3.1415926535, places=5)
        return


class LowSpeedAero(unittest.TestCase):

    def test_discretesource(self):
        """Check that the flowfield around a source makes sense."""
        # Define grid
        x = np.linspace(-2, 2, 20)
        z = x.copy()
        X, Z = np.meshgrid(x, z)

        def compute_plotting_params(xgrid, zgrid, sigma):
            """Get direction vectors Uhat and What, and velocity magnitude."""
            U, W = SORC2D(sigmaj=sigma, x=xgrid, z=zgrid, xj=0, zj=0)
            V = (U ** 2 + W ** 2) ** 0.5
            Uhat = U / V  # Normalised U velocities (x-direction)
            What = W / V  # Normalised W velocities (z-direction)
            return Uhat, What, V

        data_cw = compute_plotting_params(X, Z, sigma=1)
        data_ccw = compute_plotting_params(X, Z, sigma=-1)
        data = [data_cw, data_ccw]
        colour = ["blue", "orange"]

        fig, axs = plt.subplots(1, 2, figsize=(6, 3.3), dpi=140)
        fig.suptitle("Discrete Source Element in 2D", size="x-large")

        for i, ax in enumerate(axs.flat):
            ax.quiver(X, Z, *data[i], scale=20, headwidth=6)
            ax.streamplot(X, Z, *data[i][0:2], density=0.5, color=colour[i])
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

        return

    def test_discretevortex(self):
        """Check that the flowfield around a vortex makes sense."""
        # Define grid
        x = np.linspace(-2, 2, 20)
        z = x.copy()
        X, Z = np.meshgrid(x, z)

        def compute_plotting_params(xgrid, zgrid, Gamma):
            """Get direction vectors Uhat and What, and velocity magnitude."""
            U, W = VOR2D(Gammaj=Gamma, x=xgrid, z=zgrid, xj=0, zj=0)
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

        return


if __name__ == '__main__':
    unittest.main()
