"""Base methods for estimating drag in 2D and 3D."""
import numpy as np

from carpy.environment import LIBREF_ATM
from carpy.utility import Hint, Quantity, isNone

__all__ = ["AeroSolution"]
__author__ = "Yaseen Reza"


# ============================================================================ #
# Support Functions and Classes
# ---------------------------------------------------------------------------- #

class AeroSolution(object):
    """
    Template object of incompressible lift/drag performance coefficients, which
    solvers should aim to fill for a given angle of attack.
    """
    _CL: float = NotImplemented
    _CD0: float = NotImplemented
    _CDi: float = NotImplemented
    _CDw: float = NotImplemented
    _CD: float = NotImplemented
    _CY: float = NotImplemented
    _Cl: float = NotImplemented
    _Cm: float = NotImplemented
    _Cn: float = NotImplemented
    _Cni: float = NotImplemented
    _x_cp: float = NotImplemented

    def __init__(self, sections=None, /,  # <-- position only argument
                 altitude: Hint.num = None, TAS: Hint.num = None,
                 alpha: Hint.num = None, beta: Hint.num = None,
                 geometric: bool = None, atmosphere=None, N: int = None,
                 **kwargs):
        self._sections = sections
        self._altitude = Quantity(0.0 if altitude is None else altitude, "m")
        self._TAS = Quantity(0.0 if TAS is None else TAS, "m s^{-1}")
        self._alpha = 0.0 if alpha is None else float(alpha)
        self._beta = 0.0 if beta is None else float(beta)
        self._geometric = False if geometric is None else geometric
        self._atmosphere = LIBREF_ATM if atmosphere is None else atmosphere
        self._Nctrlpts = 40 if N is None else int(N)
        # Track which attributes are accessed during computation:
        self._init_accessed = dict()
        self._user_readable = False  # Record any accesses when this is False
        return

    def __str__(self):
        # I know the formatting below looks funky, but I promise, it works
        # Also, the ': .6F' specifier keeps a space for the sign of a -ve number
        returnstr = (
            f"""** {self._sections} **
 CD  = {self.CD: .6F}    CY  = {self.CY: .6F}    CL  = {self.CL: .6F}
 CD0 = {self.CD0: .6F}    CDi = {self.CDi: .6F}    CDw = {self.CDw: .6F}
 Cl  = {self.Cl: .6F}    Cm  = {self.Cm: .6F}    Cn  = {self.Cn: .6F}
 Cli = {np.nan: .6F}    Cmi = {np.nan: .6F}    Cni = {self.Cni: .6F}"""
        ).replace("NAN", "     NAN")
        return returnstr

    def __or__(self, other):
        """Logical OR, fills missing parameters of self with other"""
        # Verify objects are of the same type, and it makes sense to logical OR
        errormsg = f"Union only applies if both objects are of {type(self)=}"
        if AeroSolution in type(other).__bases__:
            pass  # Both self and other are children of the WingSolution class
        elif type(other) is AeroSolution:
            pass  # Self is a child of WingSolution, other *is* a WingSolution
        else:
            raise TypeError(errormsg)

        # Verify it's okay to concatenate the predictions:
        # ... are the solutions for common lift bodies?
        errormsg = (
            f"Can only apply union when wingsections attributes are identical "
            f"(actually got {self.sections, other.sections})"
        )
        if not any(isNone(self.sections, other.sections)):
            assert self.sections is other.sections, errormsg

        # ... do the solutions have compatible flight conditions?
        if not all([self._user_readable, getattr(other, "_user_readable")]):
            errormsg = (
                f"{self.__repr__()} or {other.__repr__()} did not have the "
                f"'_user_readable' flag set to true. Did you forget to set it "
                f"at the end of __init__ or any other dunder method?"
            )
            raise RuntimeError(errormsg)
        # Observing the following private attributes can change their value,
        # Schrödinger's variable style, when using an interactive debugger. For
        # your own sanity, these parameters are now saved into definite vars.
        accessed_self = self._init_accessed
        accessed_other = getattr(other, "_init_accessed")

        # If both solutions depend on a parameter, make sure they are similar
        common_attrs = set(accessed_self) & set(accessed_other)
        errormsg = (
            f"Can't do union for {type(self).__name__} objects, found that one "
            f"or more of the following values had mismatches: {common_attrs}"
        )
        for attr in common_attrs:
            if getattr(self, f"_{attr}") == getattr(other, f"_{attr}"):
                continue  # The parameters are similar, do nothing :)
            raise ValueError(errormsg)  # Uh oh, flight conditions differed!

        # Find instantiation arguments of self and other, combine them
        new_kwargs = {
            "altitude": np.nan, "TAS": np.nan,  # <-- defaults
            **{x: getattr(self, x) for x in accessed_self},
            **{x: getattr(other, x) for x in accessed_other}
        }

        # Find performance parameters of self and other, combine them
        to_combine = "CY,CL,CD0,CDi,CDw,Cl,Cm,Cn,Cni,x_cp".split(",")
        result_self = {attr: getattr(self, attr) for attr in to_combine}
        result_other = {attr: getattr(other, attr) for attr in to_combine}
        result_new = {
            attr: result_self[attr]
            if ~np.isnan(result_self[attr]) else result_other[attr]
            for attr in to_combine
        }

        # Assign new performance parameters to new object
        new_soln = AeroSolution(self.sections, **new_kwargs)
        for (k, v) in result_new.items():
            if ~np.isnan(v):
                setattr(new_soln, f"_{k}", v)

        # Finish up with the new object
        new_soln._user_readable = True

        return new_soln

    def __add__(self, other):
        """Addition, allows drag estimators to combine drag effects."""
        # Verify it's okay to add the predictions, by first concatenating them
        errormsg = f"Addition only applies if both objects are of {type(self)=}"
        try:
            new_object = self | other
        except TypeError as e:
            raise TypeError(errormsg) from e

        # Find performance parameters of self and other, combine them
        to_combine = "CD0,CDi,CDw,CD".split(",")
        result_self = {attr: getattr(self, attr) for attr in to_combine}
        result_other = {attr: getattr(other, attr) for attr in to_combine}
        result_new = {
            attr: np.nansum((result_self[attr], result_other[attr]))
            for attr in to_combine
        }

        # Assign new performance parameters to new object
        for (k, v) in result_new.items():
            if ~np.isnan(v):
                setattr(new_object, f"_{k}", v)

        return new_object

    @property
    def sections(self):
        """The geometrical sections used in this performance prediction."""
        return self._sections

    @property
    def altitude(self) -> Quantity:
        """Altitude of the solution."""
        if not self._user_readable:
            self._init_accessed["altitude"] = True
        return self._altitude

    @property
    def TAS(self) -> Quantity:
        """True airspeed of the solution."""
        if not self._user_readable:
            self._init_accessed["TAS"] = True
        return self._TAS

    @property
    def alpha(self) -> float:
        """Angle of attack of the solution."""
        if not self._user_readable:
            self._init_accessed["alpha"] = True
        return self._alpha

    @property
    def beta(self) -> float:
        """Angle of sideslip of the solution."""
        if not self._user_readable:
            self._init_accessed["beta"] = True
        return self._beta

    @property
    def geometric(self) -> bool:
        """Whether the altitude argument is geometric or geopotential."""
        if not self._user_readable:
            self._init_accessed["geometric"] = True
        return self._geometric

    @property
    def atmosphere(self):
        """Angle of sideslip of the solution."""
        if not self._user_readable:
            self._init_accessed["atmosphere"] = True
        return self._atmosphere

    @property
    def CL(self) -> float:
        """Wing coefficient of lift, CL."""
        if self._CL is NotImplemented:
            return np.nan
        return self._CL

    @CL.deleter
    def CL(self):
        self._CL = NotImplemented
        return

    @property
    def CD0(self) -> float:
        """Profile component of geometry's coefficient of drag, CD0."""
        if self._CD0 is NotImplemented:
            return np.nan
        return self._CD0

    @CD0.deleter
    def CD0(self):
        self._CD0 = NotImplemented
        return

    @property
    def CDi(self) -> float:
        """Induced component of geometry's coefficient of drag, CDi."""
        if self._CDi is NotImplemented:
            return np.nan
        return self._CDi

    @CDi.deleter
    def CDi(self):
        self._CDi = NotImplemented
        return

    @property
    def CDw(self) -> float:
        """Wave component of geometry's coefficient of drag, CDw."""
        if self._CDw is NotImplemented:
            return np.nan
        return self._CDw

    @CDw.deleter
    def CDw(self):
        self._CDw = NotImplemented
        return

    @property
    def CD(self) -> float:
        """Geometry's coefficient of drag (profile + induced + wave), CD."""
        CD_nansum = np.nansum((self.CD0, self.CDi, self.CDw))
        if self._CD is NotImplemented:
            return CD_nansum
        elif np.isclose(self._CD, CD_nansum):
            self._CD = NotImplemented
            return CD_nansum
        else:
            errormsg = (
                f"Total drag coefficient CD is not equal to the sum of "
                f"component profile, induced, and wave drags! ({self._CD} != "
                f"{self.CD0=} + {self.CDi} + {self.CDw=})"
            )
            raise ValueError(errormsg)

    @CD.deleter
    def CD(self):
        del self.CD0, self.CDi, self.CDw
        return

    @property
    def CY(self) -> float:
        """Coefficient of side/lateral force (+ve := slip right), CY."""
        if self._CY is NotImplemented:
            return np.nan
        return self._CY

    @CY.deleter
    def CY(self):
        self._CY = NotImplemented
        return

    @property
    def Cl(self) -> float:
        """Rolling moment coefficient (+ve := roll right), Cl."""
        if self._Cl is NotImplemented:
            return np.nan
        return self._Cl

    @Cl.deleter
    def Cl(self):
        self._Cl = NotImplemented
        return

    @property
    def Cm(self) -> float:
        """Pitching moment coefficient (+ve := pitch up), Cm."""
        if self._Cm is NotImplemented:
            return np.nan
        return self._Cm

    @Cm.deleter
    def Cm(self):
        self._Cm = NotImplemented
        return

    @property
    def Cn(self) -> float:
        """Yawing moment coefficient (+ve := right rudder), Cn."""
        if self._Cn is NotImplemented:
            return np.nan
        return self._Cn

    @Cn.deleter
    def Cn(self):
        self._Cn = NotImplemented
        return

    @property
    def Cni(self) -> float:
        """Induced yawing moment coefficient (+ve := right rudder), Cni."""
        if self._Cni is NotImplemented:
            return np.nan
        return self._Cni

    @Cni.deleter
    def Cni(self):
        self._Cni = NotImplemented
        return

    @property
    def x_cp(self) -> float:
        """
        Chordwise location of centre of pressure, as a fraction of the root
        chord behind the leading edge.
        """
        if self._x_cp is NotImplemented:
            return np.nan
        return self._x_cp

    @x_cp.deleter
    def x_cp(self):
        self._x_cp = NotImplemented
        return
