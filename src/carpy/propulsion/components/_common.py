"""Common methods for propulsion components."""

__all__ = ["EnergyForm"]
__author__ = "Yaseen Reza"


class Tag(object):

    def __init__(self):
        tagname = f"{type(self).__name__}"
        raise RuntimeError(f"Tag '{tagname}' is not meant to be instantiated")


# class Tags(object):
#     """
#     A class of tags to classify the type of value being conveyed. The tags in
#     this class exist as a means of unifying definitions of what a component of
#     the network understands and perceives as a valid input.
#
#     The point of a tag is that a number may represent a quantity like power, but
#     not all types of power are immediately interchangeable in a system. For
#     example:
#
#     -   A turbine may expect fluid (hydraulic) power as an input - and to
#         ensure consistency of the code's operation, a turbine must not be able
#         to accept electrical power as a valid input despite sharing the same
#         unit dimensions.
#     -   On the other hand, a method that multiplicatively applies efficiency to
#         the idealised power output of a component is fine to accept mechanical,
#         electrical, or thermal efficiencies - but must not accept process
#         efficiencies because the process of computation is more complicated.
#
#     """
#
#     @staticmethod
#     def value(tagged_value: tuple, expected_tag):
#         """
#         Extract the value from a tagged value, should it meet conditions.
#
#         Args:
#             tagged_value: Tuple in the form (value, tag).
#             expected_tag: The tag expected of the value
#
#         Returns:
#             The value, provided the tags match.
#
#         """
#
#         value, tag = tagged_value
#
#         if tag is expected_tag:
#             return value
#         else:
#             raise ValueError(f"{tagged_value} did not meet {expected_tag=}")
#
#     class Power(Tag):
#         """Classification: object represents a type of power."""
#         unit_equivalent = "W"
#
#         class Mechanical(Tag):
#             """
#             Classification: object is of mechanical power type.
#
#             Accounts for power (torque) transmitted in shafts and levers.
#             """
#             unit_equivalent = "W"
#
#         class Electrical(Tag):
#             """
#             Classification: object is of electrical power type.
#
#             Accounts for power transmitted in or used by electrical machines.
#             """
#             unit_equivalent = "W"
#
#         class Hydraulic(Tag):
#             """
#             Classification: object is of hydraulic power type.
#
#             Accounts for power transmitted through movement of fluid.
#             """
#             unit_equivalent = "W"
#
#         class Thermal(Tag):
#             """
#             Classification: object is of thermal power type.
#
#             Accounts for heating power and thermal losses.
#             """
#             unit_equivalent = "W"
#
#     class Efficiency(Tag):
#         """Classification: object represents an efficiency."""
#         unit_equivalent = ""
#
#         class Work(Tag):
#             """
#             Classification: Work-related efficiency.
#
#             Accounts for losses in power during transmission or transformation.
#             """
#             unit_equivalent = ""
#
#             class Mechanical(Tag):
#                 """
#                 Classification: object is a mechanical efficiency.
#
#                 Accounts for power loss, for example, in bearings/transmissions.
#                 """
#                 unit_equivalent = ""
#
#             class Electrical(Tag):
#                 """
#                 Classification: object is an electrical efficiency.
#
#                 Accounts for power loss, for example, in wiring/transformers.
#                 """
#                 unit_equivalent = ""
#
#             class Chemical(Tag):
#                 """
#                 Classification: object is a chemical efficiency.
#
#                 Accounts for power loss, for example, in fuel energy release.
#                 """
#                 unit_equivalent = ""
#
#         class Process(Tag):
#             """Classification: Thermodynamic process efficiency."""
#             unit_equivalent = ""
#
#             class Polytropic(Tag):
#                 """
#                 Classification: Process efficiency.
#
#                 Accounts for power loss through small-stage efficiency.
#                 """
#                 unit_equivalent = ""
#
#             class Isentropic(Tag):
#                 """
#                 Classification: Isentropic process efficiency.
#
#                 Accounts for power loss in isentropic processes
#                 """
#                 unit_equivalent = ""
#
#             # class Diabatic(object):
#             #     """
#             #     Classification: Diabatic process efficiency.
#             #
#             #     Accounts for power loss, for example, in diabatic ducts.
#             #     """
#             #     unit_equivalent = ""
#
#     class Conditions(Tag):
#         """Classifications of types of conditions applicable to fluid flow."""
#         unit_equivalent = NotImplemented
#
#         class State(Tag):
#             """Classification: object represents static conditions of fluid."""
#             unit_equivalent = NotImplemented
#
#         class MassFlowrate(Tag):
#             """Classification: object represents fluid mass flowrate."""
#             unit_equivalent = "kg s^{-1}"
#
#         class VolumetricFlowrate(Tag):
#             """Classification: object represents fluid volume flowrate."""
#             unit_equivalent = "m^{3} s^{-1}"
#
#     class RatioStateVars(Tag):
#         """Classification: Dimensionless ratio of thermodynamic state vars."""
#
#         class Pressure(Tag):
#             """Classification: object represents pressure ratio."""
#             unit_equivalent = ""
#
#         class Temperature(Tag):
#             """Classification: object represents temperature ratio."""
#             unit_equivalent = ""
#
#         class AdiabaticIndex(Tag):
#             """Classification: object represents specific heat ratio."""
#             unit_equivalent = ""


class EnergyForm(object):
    """
    References:
        -   Forms of energy, online, `eia.gov/energyexplained/what-is-energy/forms-of-energy.php`
            (Accessed 05/06/2023).
    """

    class Potential(Tag):
        """Stored energy and the energy of position."""

    # Energy in bonds
    Chemical = type("Chemical", (Potential,), dict())
    # Energy in tension
    Mechanical = type("Mechanical", (Potential,), dict())
    # Energy in nuclei
    Nuclear = type("Nuclear", (Potential,), dict())
    # Energy in work against gravity
    Gravitational = type("Gravitational", (Potential,), dict())

    class Kinetic(Tag):
        """Energy due to motion of waves, particles, substances, and objects."""

    # Energy in electromagnetic radiation
    Radiant = type("Radiant", (Kinetic,), dict())
    # Energy in movement of atoms and molecules
    Thermal = type("Thermal", (Kinetic,), dict())
    # Energy in object inertia
    Motion = type("Motion", (Kinetic,), dict())
    # Energy in compression/rarefaction waves in substances
    Sound = type("Sound", (Kinetic,), dict())
    # Energy in movement of charge
    Electrical = type("Radiant", (Kinetic,), dict())
