"""Module containing definitions of common fluids."""
from carpy.physicalchem import UnreactiveFluidModel, species, SRKmodPeneloux

__author__ = "Yaseen Reza"


def Jet_A(eos_class=None) -> UnreactiveFluidModel:
    """
    A representative model for Jet-A aviation turbine fuel.

    This model is based on the findings of Huber et al. (2010) in creating a surrogate model for Jet-A-4658, which
    is a composite mixture of several Jet-A samples from different manufacturers.

    Notes:
        Due to the infancy of the library, we do not yet have the ability to compose jet fuel from its constituent
        species. As a result, this is at best a representation of the average properties of the fluid.

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
