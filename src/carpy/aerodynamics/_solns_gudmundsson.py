"""
Methods for predicting aerodynamic performance, from Gudmundsson.

References:
        S. Gudmundsson, "General Aviation Aircraft Design: Applied Methods and
        Procedures", Butterworth-Heinemann, 2014.
"""

import numpy as np

from carpy.geometry import WingSections
from carpy.utility import Hint, Quantity, constants as co, moving_average
from ._common import AeroSolution

__all__ = ["MixedBLDrag"]
__author__ = "Yaseen Reza"


class MixedBLDrag(AeroSolution):
    """
    Method for predicting viscous drag due to skin friction effects.

    References:
        S. Gudmundsson, "General Aviation Aircraft Design: Applied Methods and
        Procedures", Butterworth-Heinemann, 2014, pp. 675-685.
    """

    def __init__(self, wingsections: WingSections,
                 Ks: Hint.num = None, CDf_CD0: Hint.num = None, **kwargs):
        # Super class call
        super().__init__(wingsections, **kwargs)

        # Recast as necessary
        Ks = co.MATERIAL.roughness_Ks.paint_matte_smooth if Ks is None else Ks
        Ks = np.mean(Ks)
        # Assume 85% of profile drag is from skin friction (15% pressure drag)
        self._CDf_CD0 = 0.85 if CDf_CD0 is None else CDf_CD0

        # Bail early if necessary (why would anyone evaluate zero speed drag?)
        if self.TAS == 0:
            self._Cf_laminar = np.nan
            self._Cf_turbulent = np.nan
            self._Cf = np.nan
            self._CDf = np.nan
            self._CD0 = self._CDf / self.CDf_CD0
            return

        # Linearly spaced sections defined in the span
        sections = wingsections.spansections(N=(N := self._Nctrlpts))

        # Step 1) Find viscosity of air
        fltconditions = dict([
            ("altitude", self.altitude), ("geometric", self.geometric)])
        mu_visc = self.atmosphere.mu_visc(**fltconditions)

        # Step 2) Compute Reynolds number
        rho = self.atmosphere.rho(**fltconditions)
        chords = Quantity([sec.chord.x for sec in sections], "m")
        Re = rho * self.TAS * chords / mu_visc

        # Step 3) Cutoff Reynolds number due to surface roughness effects
        Mach = self.TAS / self.atmosphere.c_sound(**fltconditions)
        if Mach <= 0.7:
            Re_cutoff = 38.21 * (chords / Ks).x ** 1.053
        else:
            Re_cutoff = 44.62 * (chords / Ks).x ** 1.053 * Mach ** 1.16
        Re = np.vstack((Re, Re_cutoff)).min(axis=0)

        # Step 4) Compute skin friction coefficient for fully laminar/turbulent
        self._sectionCf_laminar = 1.328 * Re ** -0.5
        self._sectionCf_turbulent = 0.455 * np.log10(Re) ** -2.58
        # Compressiblity correction
        self._sectionCf_turbulent *= (1 + 0.144 * Mach ** 2) ** -0.65

        # Step 5) Determine fictitious turbulent boundary layer origin point X0
        Xtr_C = 0.5  # Assume transition @X==0.5, avg of upper/lower transitions
        X0_C = 36.9 * Xtr_C ** 0.625 * Re ** -0.375

        # Step 6) Compute mixed laminar-turbulent flow skin friction coefficient
        # Young's method:
        Cfs = 0.074 * Re ** -0.2 * (1 - (Xtr_C - X0_C)) ** 0.8

        # Find the chords between the stations at which Cf is evaluated
        mid_chords = moving_average([sec.chord.x for sec in sections])
        # Discretise the Sref of the wing into components centred on mid_chords
        mid_Srefs = mid_chords * wingsections.b.x / (N - 1)
        mid_perim = moving_average([sec.aerofoil.perimeter for sec in sections])
        mid_Swets = mid_Srefs * mid_perim

        # Step 7) Compute skin friction drag coefficient
        Swet = mid_Swets.sum()
        Sref = mid_Srefs.sum()
        self._Cf = (moving_average(Cfs) * mid_Swets).sum() / Swet
        self._CDf = self._Cf * (Swet / Sref)

        # Assignments
        self._CD0 = self._CDf / self.CDf_CD0

        # Finish up
        self._user_readable = True

        return

    @property
    def Cf_laminar(self) -> float:
        """100% Laminar limit of skin friction coefficient, Cflaminar."""
        return self._Cf_laminar

    @property
    def Cf_turbulent(self) -> float:
        """100% Turbulent limit of skin friction coefficient, Cfturbulent."""
        return self._Cf_turbulent

    @property
    def Cf(self) -> float:
        """Skin friction coefficient, Cf."""
        return self._Cf

    @property
    def CDf(self) -> float:
        """Coefficient of friction drag, CDf."""
        return self._CDf

    @property
    def CDf_CD0(self) -> float:
        """Proportion of profile drag that is composed of skin friction drag."""
        return self._CDf_CD0

    @CDf_CD0.setter
    def CDf_CD0(self, value):
        self._CDf_CD0 = float(value)
        self._CD0 = self.CDf / self.CDf_CD0  # Update profile drag estimate
        return
