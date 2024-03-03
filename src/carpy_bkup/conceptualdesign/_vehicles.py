"""Methods for describing the aircraft type during conceptual design phase."""

__all__ = ["ConceptAeroplane"]
__author__ = "Yaseen Reza"

# ============================================================================ #
# Support functions and classes
# ---------------------------------------------------------------------------- #

# Classifications
Vehicle = type("Vehicle", (object,), {})
Aircraft = type("Aircraft", (Vehicle,), {})
FixedWing = type("FixedWing", (Aircraft,), {})
RotaryWing = type("RotaryWing", (Aircraft,), {})


class ConceptAeroplane(FixedWing):
    """
    One of many classes for carrying out the conceptual analysis of vehicles.
    This class is for fixed-wing concepts.
    """

    def __init__(self):
        raise NotImplementedError("Not ready yet! sorry")
