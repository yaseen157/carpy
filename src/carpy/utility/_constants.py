"""Assorted constants used in physics or standards."""
import numpy as np

from carpy.utility._unitconversion import Quantity

__all__ = ["constants"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Constants of the Universe
# ---------------------------------------------------------------------------- #
# Constants from Wikipedia

class Physical(object):
    """
    Assorted physical constants. Return types of this class use the Quantity
    object to describe the appropriate units for the constant's value.

    Sourced from Wikipedia's list of physical constants.
    """

    @property
    def G(self) -> Quantity:
        """Newtonian constant of gravitation"""
        return Quantity(6.67430e-11, "m^{3} kg^{-1} s^{-2}")

    @property
    def c(self) -> Quantity:
        """Speed of light in vacuum"""
        return Quantity(299_792_458, "m s^{-1}")

    @property
    def h(self) -> Quantity:
        """Planck constant"""
        return Quantity(6.626_070_15e-34, "J Hz^{-1}")

    @property
    def hbar(self) -> Quantity:
        """Reduced Planck constant"""
        return self.h / (2 * np.pi)

    @property
    def mu_0(self) -> Quantity:
        """Vacuum magnetic permeability"""
        return Quantity(1.256_637_062_12e-6, "N A^{-2}")

    @property
    def Z_0(self) -> Quantity:
        """Characteristic impedance of vacuum"""
        return self.mu_0 * self.c

    @property
    def eps_0(self) -> Quantity:
        """Vacuum electric permittivity"""
        return 1 / (self.mu_0 * self.c ** 2)

    @property
    def k_e(self) -> Quantity:
        """Coulomb constant"""
        return 1 / (4 * np.pi * self.eps_0)

    @property
    def k_B(self) -> Quantity:
        """Boltzmann constant"""
        return Quantity(1.380_649e-23, "J K^{-1}")

    @property
    def sigma(self) -> Quantity:
        """Stefan-Boltzmann constant"""
        return np.pi ** 2 * self.k_B ** 4 / (60 * self.hbar ** 3 * self.c ** 2)

    @property
    def b(self) -> Quantity:
        """Wien wavelength displacement law constant"""
        return Quantity(2.897_771_955e10 - 3, "m K")

    @property
    def bprime(self) -> Quantity:
        """Wien frequency displacement law constant"""
        return Quantity(5.878_925_757e10, "Hz K^{-1}")

    @property
    def b_entropy(self) -> Quantity:
        """Wien entropy displacement law constant"""
        return Quantity(3.002_916_077, "m K")

    @property
    def e(self) -> Quantity:
        """Elementary charge"""
        return Quantity(1.602_176_634e-19, "C")

    @property
    def G_0(self) -> Quantity:
        """Conductance quantum"""
        return 2 * self.e ** 2 / self.h

    @property
    def R_K(self) -> Quantity:
        """von Klitzing constant"""
        return self.h * self.e ** -2

    @property
    def K_J(self) -> Quantity:
        """Josephson constant"""
        return 2 * self.e * self.h ** -1

    @property
    def Phi_0(self) -> Quantity:
        """Magnetic flux quantum"""
        return self.h / (2 * self.e)

    @property
    def alpha(self) -> Quantity:
        """Fine-structure constant"""
        return self.e ** 2 / (4 * np.pi * self.eps_0 * self.hbar * self.c)

    @property
    def m_e(self) -> Quantity:
        """Electron, proton, and neutron mass"""
        return Quantity(9.109_383_7015e-31, "kg")

    @property
    def m_p(self) -> Quantity:
        """Electron, proton, and neutron mass"""
        return Quantity(1.672_621_923_69e-27, "kg")

    @property
    def m_n(self) -> Quantity:
        """Electron, proton, and neutron mass"""
        return Quantity(1.674_927_498_04e-27, "kg")

    @property
    def m_mu(self) -> Quantity:
        """Muon mass"""
        return Quantity(1.883_531_627e-28, "kg")

    @property
    def m_tau(self) -> Quantity:
        """Tau particle mass"""
        return Quantity(3.167_54e-27, "kg")

    @property
    def m_t(self) -> Quantity:
        """Top quark mass"""
        return Quantity(3.0784e-25, "kg")

    @property
    def g_e(self) -> Quantity:
        """Electron g-factor"""
        return Quantity(-2.002_319_304_362_56)

    @property
    def g_mu(self) -> Quantity:
        """Muon g-factor"""
        return Quantity(-2.002_331_8418)

    @property
    def g_p(self) -> Quantity:
        """Proton g-factor"""
        return Quantity(5.585_694_6893)

    @property
    def mu_B(self) -> Quantity:
        """Bohr magneton"""
        return self.e * self.hbar / (2 * self.m_e)

    @property
    def mu_N(self) -> Quantity:
        """Nuclear magneton"""
        return self.e * self.hbar / (2 * self.m_p)

    @property
    def r_e(self) -> Quantity:
        """Classical electron radius"""
        return self.e ** 2 * self.k_e / (self.m_e * self.c ** 2)

    @property
    def sigma_e(self) -> Quantity:
        """Thomson cross-section"""
        return (8 * np.pi / 3) * self.r_e ** 2

    @property
    def a_0(self) -> Quantity:
        """Bohr radius"""
        return self.hbar ** 2 / (self.k_e * self.m_e * self.e ** 2)

    @property
    def E_h(self) -> Quantity:
        """Hartree energy"""
        return self.alpha ** 2 * self.c ** 2 * self.m_e

    @property
    def Ry(self) -> Quantity:
        """Rydberg unit of energy"""
        return self.E_h / 2

    @property
    def R_infty(self) -> Quantity:
        """Rydberg constant"""
        return self.Ry / (self.h * self.c)

    @property
    def N_A(self) -> Quantity:
        """Avogadro constant"""
        return Quantity(6.022_140_76e23, "mol^{-1}")

    @property
    def R(self) -> Quantity:
        """Molar gas constant"""
        return self.N_A * self.k_B

    @property
    def F(self) -> Quantity:
        """Faraday constant"""
        return self.N_A * self.e

    @property
    def m_12C(self) -> Quantity:
        """Atomic mass of carbon-12"""
        return Quantity(1.992_646_879_92e-26, "kg")

    @property
    def M_12C(self) -> Quantity:
        """Molar mass of carbon-12"""
        return Quantity(11.999_999_9958e-3, "kg mol^{-1}")

    @property
    def m_u(self) -> Quantity:
        """Atomic mass constant, one-twelfth the mass of carbon-12"""
        return self.m_12C / 12

    @property
    def M_u(self) -> Quantity:
        """Molar mass constant, one-twelfth the molar mass of carbon-12"""
        return self.M_12C / 12


# ============================================================================ #
# Constants defined in Standards
# ---------------------------------------------------------------------------- #
# Constants from World Geodetic System

class WGS84(object):
    """
    Assorted constants and parameters derived from, or used in the construction
    of, the World Geodetic System (1984).
    """

    @property
    def a(self) -> Quantity:
        """Semi-major axis of the world ellipsoid"""
        return Quantity(6_378_137, "m")

    @property
    def omega(self) -> Quantity:
        """Angular velocity"""
        return Quantity(7.292115e-5, "rad s^{-1}")

    @property
    def GM(self) -> Quantity:
        """Geocentric gravitational constant (including mass of atmosphere)"""
        return Quantity(398600.5, "km^{3} s^{-2}")

    @property
    def Cbar2_0(self) -> Quantity:
        """
        Normalised second degree zonal harmonic coefficient of the gravitational
        potential.
        """
        return Quantity(-484.16685e-6)

    @property
    def inv_f(self) -> Quantity:
        """Flattening parameter of the world ellipsoid"""
        return Quantity(1 / 298.257_223_563)

    @property
    def e(self) -> Quantity:
        """Eccentricity parameter of the world ellipsoid"""
        return (self.inv_f * (2 - self.inv_f)) ** 0.5

    @property
    def b(self) -> Quantity:
        """Ellipsoidal polar radius"""
        return self.a * (1 - self.e ** 2) ** 0.5

    @property
    def r45(self) -> Quantity:
        """Radius at +45deg latitude on the world ellipsoid"""
        cos45squared = 0.5
        return (self.b ** 2 / (1 - self.e ** 2 * cos45squared)) ** 0.5


# ---------------------------------------------------------------------------- #
# Constants from ISO 2533 (1975)

class StandardSealevel(object):
    """
    A class organising standards of mean sea level conditions used in the ISO
    Standard Atmosphere, as it is described in ISO publication 2533:1975 first
    edition.
    """

    @property
    def g(self) -> Quantity:
        """Standard acceleration of free fall at 45deg 32' 33''"""
        return Quantity(9.80665, "m s^{-2}")

    @property
    def M(self) -> Quantity:
        """Air molar mass at sea level as obtained from the perfect gas law"""
        return Quantity(28.964_420, "kg kmol^{-1}")

    @property
    def p(self) -> Quantity:
        """Standard air pressure"""
        return Quantity(101.325e3, "Pa")

    @property
    def Rs(self) -> Quantity:
        """Specific gas constant"""
        return Quantity(287.052_87, "J K^{-1} kg^{-1}")

    @property
    def To(self) -> Quantity:
        """Thermodynamic ice-point temperature, at mean sea level"""
        return Quantity(273.15, "K")

    @property
    def T(self) -> Quantity:
        """Standard thermodynamic air temperature at mean sea level"""
        return Quantity(288.15, "K")

    @property
    def rho(self) -> Quantity:
        """Standard air density"""
        return Quantity(1.225, "kg m^{-3}")

    @property
    def gamma(self) -> Quantity:
        """
        Adiabatic index, the ratio of specific heat of air at constant pressure
        to its specific heat at constant volume.
        """
        return Quantity(1.4)

    @property
    def sigma(self) -> Quantity:
        """
        Effective collosion diameter of an air molecule, taken as constant with
        altitude.
        """
        return Quantity(0.365e-9, "m")

    @property
    def X(self) -> dict:
        """
        Dry, clean air composition at sea level.

        Because Ozone (O3), Sulphur dioxide (SO2), Nitrogen dioxide (NO2) and
        Iodine (I2) quantities can vary from time to time or place to place,
        their contributions are omitted.
        """
        composition = dict([
            ("N2", 78.084), ("O2", 20.947_6), ("Ar", 0.934), ("CO2", 0.031_4),
            ("Ne", 1.818e-3), ("He", 524.0e-6), ("Kr", 114.0e-6),
            ("Xe", 8.7e-6), ("H2", 50.0e-6), ("N2O", 50.0e-6), ("CH4", 0.2e-3)
        ])
        return composition


class ISO2533(StandardSealevel):
    """
    A class organising standards used in the ISO Standard Atmosphere, as it is
    described in ISO publication 2533:1975 first edition.
    """

    @property
    def NA(self) -> Quantity:
        """Avogardro's constant, as adopted in 1961 by IUPAC"""
        return Quantity(602.257e24, "kmol^{-1}")

    @property
    def R(self) -> Quantity:
        """Universal gas constant"""
        return Quantity(8_314.32, "J K^{-1} kmol^{-1}")

    @property
    def S(self) -> Quantity:
        """
        Sutherland empirical coefficient S in equation for dynamic viscosity.
        """
        return Quantity(110.4, "K")

    @property
    def betaS(self) -> Quantity:
        """
        Sutherland empirical coefficient beta in equation for dynamic viscosity.
        """
        return Quantity(1.458e-6, "kg m^{-1} s^{-1} K^{-0.5}")

    @property
    def r(self) -> Quantity:
        """Nominal Earth's radius"""
        return Quantity(6_356_766, "m")


# ---------------------------------------------------------------------------- #
# Constants from the joint Astronautical Almanac


class AA21K(object):
    """
    Astronautical Almanac (2021) section K, as described by HM Nautical Almanac
    Office and United States National Observatory.

    Notes:
        https://aa.usno.navy.mil/downloads/publications/Constants_2021.pdf

    """

    @property
    def GM_E(self) -> dict:
        """
        Gravitational parameter of Earth, a.k.a. geocentric gravitational
        constant.

        Returns:
            Geocentric gravitational constants for TCB (Barycentric Coordinate
                Time), TT (Terrestrial Time), TDB (Barycentric Dynamical Time).

        """
        GM_E = {
            "TCB": Quantity(3.986_004_418e14, "m^{3} s^{-2}"),
            "TT": Quantity(3.986_004_415e14, "m^{3} s^{-2}"),
            "TDB": Quantity(3.986_004_356e14, "m^{3} s^{-2}")
        }
        return GM_E

    @property
    def M_E(self) -> Quantity:
        """Mass of the Earth"""
        return Quantity(5.97_22e24, "kg")


# ---------------------------------------------------------------------------- #

class Standards(object):
    """A class structure for organising constants based on standards"""
    _aa21k = AA21K()
    _iso2553 = ISO2533()
    _ssl = StandardSealevel()
    _wgs84 = WGS84()

    @property
    def AA21_K(self) -> AA21K:
        """Constants defined in the Astronautical Almanac, Section K, 2021."""
        return self._aa21k

    @property
    def ISO_2533(self) -> ISO2533:
        """Conditions of the International Standard Atmosphere per ISO 2553."""
        return self._iso2553

    @property
    def SL(self) -> StandardSealevel:
        """Conditions at sea level in the standard atmosphere per ISO 2553."""
        return self._ssl

    @property
    def WGS84(self) -> WGS84:
        """Parameters of the World Geodetic System 1984."""
        return self._wgs84


# ============================================================================ #
# Useful guesses for values in materials, from wherever on the internet
# ---------------------------------------------------------------------------- #

class RoughnessKs(object):
    """
    References:
        -   DATCOM Table 4.1.5.1-A
        -   S. Gudmundsson General Aviation Aircraft Design (methods on
            estimating skin friction).
        -   University of Texas at Austin, online: https://www.caee.utexas.edu/prof/kinnas/319lab/notes13/table10.4.pdf
    """

    @property
    def composite_molded(self) -> Quantity:
        """Smooth, molded composite surface."""
        return Quantity(0.518, "um")

    @property
    def concrete(self) -> Quantity:
        """Concrete."""
        return Quantity([0.3, 3], "mm")

    @property
    def glass_smooth(self) -> Quantity:
        """Smooth glass surface."""
        return Quantity(0, "m")

    @property
    def iron_cast_asphalted(self) -> Quantity:
        """Asphalted cast iron."""
        return Quantity(0.12, "mm")

    @property
    def iron_cast(self) -> Quantity:
        """Production-level quality of cast iron surface."""
        return Quantity(254, "um")

    @property
    def iron_galvanised(self) -> Quantity:
        """Galvanised iron."""
        return Quantity(0.15, "mm")

    @property
    def iron_wrought(self) -> Quantity:
        """Wrought (worked) iron/steel."""
        return Quantity(0.046, "mm")

    @property
    def metal_galvanised(self) -> Quantity:
        """Dip galvanised metal."""
        return Quantity(152.4, "um")

    @property
    def metal_smooth(self) -> Quantity:
        """Carefully polished metal surface."""
        return Quantity([0.508, 2.032], "um")

    @property
    def paint_camo_smooth(self) -> Quantity:
        """Carefully applied camouflage paint on aluminium."""
        return Quantity(10.16, "um")

    @property
    def paint_camo(self) -> Quantity:
        """Production-level quality application of camouflage paint."""
        return Quantity(30.48, "um")

    @property
    def paint_matte_smooth(self) -> Quantity:
        """Careful (smooth) application of matte paint."""
        return Quantity(6.350, "um")

    @property
    def plastic_smooth(self) -> Quantity:
        """Smooth plastic surface."""
        return Quantity(0, "m")

    @property
    def sheetmetal_smooth(self) -> Quantity:
        """Carefully polished sheet metal surface."""
        return Quantity(1.524, "um")

    @property
    def sheetmetal(self) -> Quantity:
        """Production-level quality of sheet metal surface."""
        return Quantity(4.064, "um")

    @property
    def steel_riveted(self) -> Quantity:
        """Riveted steel."""
        return Quantity([0.9, 9], "mm")

    @property
    def steel_wrought(self) -> Quantity:
        """Wrought (worked) iron/steel."""
        return Quantity(0.046, "mm")

    @property
    def wood_smooth(self) -> Quantity:
        """Carefully polished wooden surface."""
        return Quantity([0.508, 2.032], "um")


# ---------------------------------------------------------------------------- #

class Material(object):
    """Library of aeronautical material references"""

    _roughness_Ks = RoughnessKs()

    @property
    def roughness_Ks(self) -> RoughnessKs:
        """
        Returns a dictionary of representative values for hydrodynamic
        surface roughness (equivalent sand-grain roughness, Ks).
        """
        return self._roughness_Ks


# ============================================================================ #
class Constants(object):
    """A class structure for organising assortments of constants"""

    _material = Material()
    _physical = Physical()
    _standard = Standards()

    @property
    def MATERIAL(self) -> Material:
        """A library of assorted reference material properties."""
        return self._material

    @property
    def PHYSICAL(self) -> Physical:
        """Physical constants of the Universe."""
        return self._physical

    @property
    def STANDARD(self) -> Standards:
        """Constants defined by standards."""
        return self._standard


# Allow the below object to be imported and used by users
constants = Constants()
