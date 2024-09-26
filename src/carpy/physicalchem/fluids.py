"""Module containing definitions of common fluids."""
from carpy.physicalchem import UnreactiveFluidModel, species, SRKmodPeneloux

__author__ = "Yaseen Reza"


def Jet_A_3638(eos_class=None) -> UnreactiveFluidModel:
    """
    A representative model for Jet-A aviation turbine fuel.

    This model is based on the findings of Huber et al. (2010) in creating the Jet-A surrogate 'Jet-A-3638', which
    is a composite mixture of several Jet-A samples from different manufacturers.

    References:
        Huber, M.L., Lemmon, E.W. and Bruno, T.J., 2010. Surrogate mixture models for the thermophysical properties
        of aviation fuel Jet-A. Energy & Fuels, 24(6), pp.3565-3571. https://doi.org/10.1021/ef100208c

    """
    if eos_class is None:
        eos_class = SRKmodPeneloux

    fluid_model = UnreactiveFluidModel(eos_class=eos_class)
    fluid_model.X = {
        species.n_hexylcyclohexane(): 0.268,
        species._1_methyldecalin(): 0.064,
        species._5_methylnonane(): 0.130,
        species._2_methyldecane(): 0.284,
        species.n_tetradecane(): 0.035,
        species.ortho_xylene(): 0.094,
        species.tetralin(): 0.125
    }
    return fluid_model


def Jet_A_4658(eos_class=None) -> UnreactiveFluidModel:
    """
    A representative model for Jet-A aviation turbine fuel.

    This model is based on the findings of Huber et al. (2010) in creating the Jet-A surrogate 'Jet-A-4658', which
    is a composite mixture of several Jet-A samples from different manufacturers.

    References:
        Huber, M.L., Lemmon, E.W. and Bruno, T.J., 2010. Surrogate mixture models for the thermophysical properties
        of aviation fuel Jet-A. Energy & Fuels, 24(6), pp.3565-3571. https://doi.org/10.1021/ef100208c

    """
    if eos_class is None:
        eos_class = SRKmodPeneloux

    fluid_model = UnreactiveFluidModel(eos_class=eos_class)
    fluid_model.X = {
        species.n_heptylcyclohexane(): 0.279,
        species._1_methyldecalin(): 0.013,
        species._5_methylnonane(): 0.165,
        species._2_methyldecane(): 0.154,
        species.n_tetradecane(): 0.057,
        species.n_hexadecane(): 0.033,
        species.ortho_xylene(): 0.071,
        species.tetralin(): 0.228
    }
    return fluid_model


def RP_1(eos_class=None) -> UnreactiveFluidModel:
    """
    A representative model for RP-1 rocket fuel.

    References:
        Huber, M.L., Lemmon, E.W., Ott, L.S. and Bruno, T.J., 2009. Preliminary surrogate mixture models for the
        thermophysical properties of rocket propellants RP-1 and RP-2. Energy & fuels, 23(6), pp.3083-3088.


    """
    if eos_class is None:
        eos_class = SRKmodPeneloux

    fluid_model = UnreactiveFluidModel(eos_class=eos_class)
    fluid_model.X = {
        species._1_methyldecalin(): 0.354,  # <-- Unsure about this, we want alpha-methyldecalin
        species._5_methylnonane(): 0.150,
        species.n_dodecane(): 0.183,
        species.n_heptylcyclohexane(): 0.313
    }
    return fluid_model


def RP_2(eos_class=None) -> UnreactiveFluidModel:
    """
    A representative model for RP-2 rocket fuel.

    References:
        Huber, M.L., Lemmon, E.W., Ott, L.S. and Bruno, T.J., 2009. Preliminary surrogate mixture models for the
        thermophysical properties of rocket propellants RP-1 and RP-2. Energy & fuels, 23(6), pp.3083-3088.


    """
    if eos_class is None:
        eos_class = SRKmodPeneloux

    fluid_model = UnreactiveFluidModel(eos_class=eos_class)
    fluid_model.X = {
        species._1_methyldecalin(): 0.354,  # <-- Unsure about this, we want alpha-methyldecalin
        species._5_methylnonane(): 0.084,
        species._2_4_dimethylnonane(): 0.071,
        species.n_dodecane(): 0.158,
        species.n_heptylcyclohexane(): 0.333
    }
    return fluid_model


def S_8(eos_class=None) -> UnreactiveFluidModel:
    """
    A representative model for S-8 synthetic aviation fuel.

    References:
        Huber, M.L., Smith, B.L., Ott, L.S. and Bruno, T.J., 2008. Surrogate mixture model for the thermophysical
        properties of synthetic aviation fuel S-8: Explicit application of the advanced distillation curve. Energy &
        Fuels, 22(2), pp.1104-1114.

    """
    if eos_class is None:
        eos_class = SRKmodPeneloux

    fluid_model = UnreactiveFluidModel(eos_class=eos_class)
    fluid_model.X = {
        species.n_nonane(): 0.03,
        species._2_6_dimethyloctane(): 0.28,
        species._3_methyldecane(): 0.34,
        species.n_tridecane(): 0.13,
        species.n_tetradecane(): 0.20,
        species.n_pentadecane(): 0.015,
        species.n_hexadecane(): 0.005
    }
    return fluid_model
