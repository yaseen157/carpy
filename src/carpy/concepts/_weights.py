"""Methods of this class support aircraft weight estimation methods."""
import re

import numpy as np

__all__ = ["Weights"]
__author__ = "Yaseen Reza"


def strict_type(_type: type, /, value):
    """Return the argument value, only if it is of type '_type'."""
    if isinstance(value, _type):
        return value
    errormsg = f"Expected type({_type.__name__}), actually got {type(value)=}"
    raise TypeError(errormsg)
    # try:
    #     casted_value = _type(value)
    # except Exception as e:
    #     raise TypeError(errormsg) from e


class StrictAttributes:
    """Base class for classes with strict requirements on attribute control."""

    def _get_properties(self) -> list:
        """Produce a list of the class' own properties"""
        my_properties = [
            x for x in dir(self.__class__)
            if isinstance(getattr(self.__class__, x), property)
        ]
        return my_properties

    def __str__(self):
        properties = self._get_properties()
        unsettable = []  # Properties that can't be set
        unreadable = []  # Properties that can't be read
        unassigned = []  # Properties that haven't been assigned
        assigned = []  # Else (theoretically regular, good ol' properties)

        for property_name in properties:

            # Unsettable properties are missing fset
            if getattr(self.__class__, property_name).fset is None:
                unsettable.append(property_name)
                continue

            # Unreadable properties will raise some sort of exception
            try:
                property_val = getattr(self, property_name)
            except BaseException:
                unreadable.append(property_name)
                continue

            # Unassigned properties are NaN or None
            if isinstance(property_val, type(None)) or np.isnan(property_val):
                unassigned.append((property_name, property_val))
                continue

            # Everything else has an assignment
            assigned.append((property_name, property_val))

        string = f"<{type(self).__name__} object @ {hex(id(self))}>"

        def extract_line(docstring):
            """Extract the first line of a docstring."""
            # The below line will cause an error if you do not document the
            # properties (attributes) of your children classes properly!
            mystring = re.sub(
                r"\n\s+", " ", re_delimiter.split(docstring)[0].lstrip("\n "))
            if mystring[-1] not in "!\".?":
                mystring += "."
            return mystring

        # Look for full stops followed by a newline or end of string character
        re_delimiter = re.compile(r"\.($|\n)")

        key = None
        try:
            for (key, val) in assigned:
                string += f"\n |-- {key:<18s} := {str(val):<8.7s} - "  # NB: |--
                doc = getattr(self.__class__, key).__doc__
                string += extract_line(doc)

            for (key, val) in unassigned:
                string += f"\n |-x {key:<18s} := {str(val):<8.7s} - "  # NB: |-x
                doc = getattr(self.__class__, key).__doc__
                string += extract_line(doc)

        except TypeError as e:
            if key is not None:
                errormsg = f"Did you properly document the '{key}' property?"
                raise AttributeError(errormsg) from e
            raise

        return string

    def __setattr__(self, key, value):
        errormsg = f"'{key}' cannot be set"
        if key.lstrip("_") in self._get_properties():
            errormsg += ", its value is write-protected (and may be inherited)"
            try:
                return super().__setattr__(key, value)
            except AttributeError as e:
                raise AttributeError(errormsg) from e
        elif key.startswith("_"):
            return super().__setattr__(key, value)
        else:
            errormsg += ", it is an unrecognised attribute"
            raise AttributeError(errormsg)


class GroupMethods(StrictAttributes):
    """Base class for the Python classes used to model weight "groups"."""

    def nansum(self):
        """
        Call to sum the values of all components in the group.

        Returns:
            Floating point sum of all component weights in the group, in kg.

        """
        weights = [0] + [  # Use default sum of zero to skip undefined error
            getattr(self, x) for x in self._get_properties()
            if getattr(self, x) is not None
        ]
        summation = np.nansum(weights)
        return summation


class StructuresGroup(GroupMethods):
    _wing = np.nan
    _tail_horizontal = np.nan
    _tail_vertical = np.nan
    _tail_ventral = np.nan
    _canard = np.nan
    _fuselage = np.nan
    _landingGear_main = np.nan
    _landingGear_nose = np.nan
    _landingGear_other = np.nan
    _alightingGear_catapult = np.nan
    _engine_mounts = np.nan
    _firewall = np.nan
    _engine_section = np.nan
    _airInduction = np.nan

    @property
    def wing(self):
        """Main wing weight, in kg."""
        return self._wing

    @wing.setter
    def wing(self, value):
        self._wing = value

    @property
    def tail_horizontal(self):
        """Horizontal tail weight, in kg."""
        return self._tail_horizontal

    @tail_horizontal.setter
    def tail_horizontal(self, value):
        self._tail_horizontal = value

    @property
    def tail_vertical(self):
        """Vertical tail weight, in kg."""
        return self._tail_vertical

    @tail_vertical.setter
    def tail_vertical(self, value):
        self._tail_vertical = value

    @property
    def tail_ventral(self):
        """Ventral tail weight, in kg."""
        return self._tail_ventral

    @tail_ventral.setter
    def tail_ventral(self, value):
        self._tail_ventral = value

    @property
    def canard(self):
        """Canard weight, in kg."""
        return self._canard

    @canard.setter
    def canard(self, value):
        self._canard = value

    @property
    def fuselage(self):
        """Fuselage weight, in kg."""
        return self._fuselage

    @fuselage.setter
    def fuselage(self, value):
        self._fuselage = value

    @property
    def landingGear_main(self):
        """Main landing gear weight, in kg."""
        return self._landingGear_main

    @landingGear_main.setter
    def landingGear_main(self, value):
        self._landingGear_main = value

    @property
    def landingGear_nose(self):
        """Nose landing gear weight, in kg."""
        return self._landingGear_nose

    @landingGear_nose.setter
    def landingGear_nose(self, value):
        self._landingGear_nose = value

    @property
    def landingGear_other(self):
        """Auxiliary landing gear weight, in kg."""
        return self._landingGear_other

    @landingGear_other.setter
    def landingGear_other(self, value):
        self._landingGear_other = value

    @property
    def engine_mounts(self):
        """The weight of the engine attachment pylons or trusses, in kg."""
        return self._engine_mounts

    @engine_mounts.setter
    def engine_mounts(self, value):
        self._engine_mounts = value

    @property
    def firewall(self):
        """
        Weight of the fireproof bulkhead compartmentalising the engine, in kg.
        """
        return self._firewall

    @firewall.setter
    def firewall(self, value):
        self._firewall = value

    @property
    def engine_section(self):
        """The weight of the nacelle(s) shrouding the engine(s), in kg."""
        return self._engine_section

    @engine_section.setter
    def engine_section(self, value):
        self._engine_section = value

    @property
    def airInduction(self):
        """Engine air-intake/duct system weight, in kg."""
        return self._airInduction

    @airInduction.setter
    def airInduction(self, value):
        self._airInduction = value


class PropulsionGroup(GroupMethods):
    _engines_installed = np.nan
    _accessoryDrive = np.nan
    _exhaustSystem = np.nan
    _engine_cooling = np.nan
    _oil_cooling = np.nan
    _engine_controls = np.nan
    _startingSystem = np.nan
    _fuel_system = np.nan
    _fuel_trapped = np.nan
    _engine_oil = np.nan

    @property
    def engines_installed(self):
        """Weight of engine parts strictly responsible for thrust, in kg."""
        return self._engines_installed

    @engines_installed.setter
    def engines_installed(self, value):
        self._engines_installed = value

    @property
    def accessoryDrive(self):
        """
        Weight of the accessory drive used to interface the engine, APU, and
        other hardware, in kg.
        """
        return self._accessoryDrive

    @accessoryDrive.setter
    def accessoryDrive(self, value):
        self._accessoryDrive = value

    @property
    def exhaustSystem(self):
        """Weight of exhaust ducting from the installed engines, in kg."""
        return self._exhaustSystem

    @exhaustSystem.setter
    def exhaustSystem(self, value):
        self._exhaustSystem = value

    @property
    def engine_cooling(self):
        """Weight of the shroud and ducting required to cool engines, in kg."""
        return self._engine_cooling

    @engine_cooling.setter
    def engine_cooling(self, value):
        self._engine_cooling = value

    @property
    def oil_cooling(self):
        """Weight of the (lubricating) oil cooling system, in kg."""
        return self._oil_cooling

    @oil_cooling.setter
    def oil_cooling(self, value):
        self._oil_cooling = value

    @property
    def engine_controls(self):
        """Weight of hardware responsible for governing the engine, in kg."""
        return self._engine_controls

    @engine_controls.setter
    def engine_controls(self, value):
        self._engine_controls = value

    @property
    def startingSystem(self):
        """Weight of the (pneumatic) engine starting system, in kg."""
        return self._startingSystem

    @startingSystem.setter
    def startingSystem(self, value):
        self._startingSystem = value

    @property
    def fuel_system(self):
        """
        Weight of the complete fuel system and tanks (minus hardware for
        in-flight refueling), in kg.
        """
        return self._fuel_system

    @fuel_system.setter
    def fuel_system(self, value):
        self._fuel_system = value

    @property
    def fuel_trapped(self):
        """Weight of ullage fuel trapped in fuel system feeds, etc., in kg."""
        return self._fuel_trapped

    @fuel_trapped.setter
    def fuel_trapped(self, value):
        self._fuel_trapped = value

    @property
    def engine_oil(self):
        """Weight of lubricating and utility oil, in kg."""
        return self._engine_oil

    @engine_oil.setter
    def engine_oil(self, value):
        self._engine_oil = value


class EquipmentGroup(GroupMethods):
    _flightControls = np.nan
    _APU = np.nan
    _instruments = np.nan
    _hydraulics = np.nan
    _pneumatics = np.nan
    _electrics = np.nan
    _avionics = np.nan
    _armament = np.nan
    _furnishings = np.nan
    _groundHandling = np.nan
    _airConditioning = np.nan
    _antiIcing = np.nan
    _photographic = np.nan
    _misc_e = 0.0

    @property
    def flightControls(self):
        """
        Weight of flight control hardware (sticks, pulleys, primary hydraulic
        system, etc.), in kg.
        """
        return self._flightControls

    @flightControls.setter
    def flightControls(self, value):
        self._flightControls = value

    @property
    def APU(self):
        """Weight of the auxiliary power unit, in kg."""
        return self._APU

    @APU.setter
    def APU(self, value):
        self._APU = value

    @property
    def instruments(self):
        """Weight of the flight instrument display, in kg."""
        return self._instruments

    @instruments.setter
    def instruments(self, value):
        self._instruments = value

    @property
    def hydraulics(self):
        """Installed hydraulic system weight, in kg."""
        return self._hydraulics

    @hydraulics.setter
    def hydraulics(self, value):
        self._hydraulics = value

    @property
    def pneumatics(self):
        """Installed pneumatic system weight, in kg."""
        return self._pneumatics

    @pneumatics.setter
    def pneumatics(self, value):
        self._pneumatics = value

    @property
    def electrics(self):
        """Installed electric system weight, in kg."""
        return self._electrics

    @electrics.setter
    def electrics(self, value):
        self._electrics = value

    @property
    def avionics(self):
        """Installed avionics system weight, in kg."""
        return self._avionics

    @avionics.setter
    def avionics(self, value):
        self._avionics = value

    @property
    def armament(self):
        """Non-removable (fixed) armaments including cannons, in kg."""
        return self._armament

    @armament.setter
    def armament(self, value):
        self._armament = value

    @property
    def furnishings(self):
        """Weight of fixed furnishings (including seats), in kg."""
        return self._furnishings

    @furnishings.setter
    def furnishings(self, value):
        self._furnishings = value

    @property
    def airConditioning(self):
        """Air conditioning/environmental control system weight, in kg."""
        return self._airConditioning

    @airConditioning.setter
    def airConditioning(self, value):
        self._airConditioning = value

    @property
    def antiIcing(self):
        """Anti-icing system weight, in kg."""
        return self._antiIcing

    @antiIcing.setter
    def antiIcing(self, value):
        self._antiIcing = value

    @property
    def photographic(self):
        """Observation payload weight, in kg."""
        return self._photographic

    @photographic.setter
    def photographic(self, value):
        self._photographic = value

    @property
    def groundHandling(self):
        """Weight of installed equipment needed for loading/handling, in kg."""
        return self._groundHandling

    @groundHandling.setter
    def groundHandling(self, value):
        self._groundHandling = value

    @property
    def misc_fixed(self):
        """Additive penalty for fixed, miscellaneous equipment, in kg."""
        return self._misc_e

    @misc_fixed.setter
    def misc_fixed(self, value):
        self._misc_e = value


class UsefulLoadGroup(GroupMethods):
    _crew = np.nan
    _fuel_usable = np.nan
    _passengers = np.nan
    _payload = np.nan
    _ammunition = np.nan
    _pylons = np.nan
    _expendableWeapons = np.nan
    _countermeasures = np.nan
    _misc_u = 0.0

    @property
    def crew(self):
        """Crew weight, in kg."""
        return self._crew

    @crew.setter
    def crew(self, value):
        self._crew = value

    @property
    def fuel_usable(self):
        """Weight of fuel that may be entirely exhausted, in kg."""
        return self._fuel_usable

    @fuel_usable.setter
    def fuel_usable(self, value):
        self._fuel_usable = value

    @property
    def passengers(self):
        """Combined weight of all boarded passengers, in kg."""
        return self._passengers

    @passengers.setter
    def passengers(self, value):
        self._passengers = value

    @property
    def payload(self):
        """Weight of payload and/or cargo, in kg."""
        return self._payload

    @payload.setter
    def payload(self, value):
        self._payload = value

    @property
    def ammunition(self):
        """Weight of ammunition onboard and used in installed guns, in kg."""
        return self._ammunition

    @ammunition.setter
    def ammunition(self, value):
        self._ammunition = value

    @property
    def countermeasures(self):
        """Weight of dispensed countermeasures e.g. chaff and flares, in kg."""
        return self._countermeasures

    @countermeasures.setter
    def countermeasures(self, value):
        self._countermeasures = value

    @property
    def misc_operator(self):
        """Additive penalty for operating equipment, in kg."""
        return self._misc_u

    @misc_operator.setter
    def misc_operator(self, value):
        self._misc_u = value


class Weights(StrictAttributes):
    """Main weight estimation class, supporting weight contributing "groups"."""

    def __init__(self, __concept, /):
        self.__concept = __concept
        self._group_structures = StructuresGroup()
        self._group_propulsion = PropulsionGroup()
        self._group_equipment = EquipmentGroup()
        self._group_usefulload = UsefulLoadGroup()
        return

    def __str__(self):
        string = f"{type(self).__name__}(concept={self._concept}, ...)"
        string += f"\n |-- W_struct = {self.structuresGroup.nansum():6.0f} [kg]"
        string += f"\n |-- W_prplsn = {self.propulsionGroup.nansum():6.0f} [kg]"
        string += f"\n |-- W_eqpmnt = {self.equipmentGroup.nansum():6.0f} [kg]"
        string += f"\n |-- W_useful = {self.usefulLoadGroup.nansum():6.0f} [kg]"
        string += f"  +\n |----------------------------"
        string += f"\n |--- W_GROSS = {self.W_designGross:6.0f} [kg]"
        return string

    @property
    def _concept(self):
        """Parent vehicle concept."""
        return self.__concept

    @property
    def structuresGroup(self) -> StructuresGroup:
        """Weights as categorised by the aircraft structural group."""
        return self._group_structures

    @property
    def propulsionGroup(self) -> PropulsionGroup:
        """Weights as categorised by the aircraft propulsion group."""
        return self._group_propulsion

    @property
    def equipmentGroup(self):
        """Weights as categorised by the aircraft equipment group."""
        return self._group_equipment

    @property
    def usefulLoadGroup(self) -> UsefulLoadGroup:
        """Weights as categorised by the aircraft useful load group."""
        return self._group_usefulload

    @property
    def W_basicEmpty(self):
        """
        Aircraft weight that consists of all items in the definition of
        "Standard Empty Weight," plus any optional and special equipment
        installed for a carrier/operator.

        Returns:
            Basic Empty Weight, in kg.

        References:
            General Aviation Manufacturer's Association.

        """
        weight = np.nansum((
            self.structuresGroup.nansum(),
            self.propulsionGroup.nansum(),
            self.equipmentGroup.nansum()
        ))
        return weight

    @property
    def W_basicOperating(self):
        """Operating empty weight, in kg."""
        summation = np.nansum((
            self.W_basicEmpty,
            self.usefulLoadGroup.crew,
            self.usefulLoadGroup.misc_operator
        ))
        return summation

    W_BOW = W_basicOperating
    W_OEW = W_basicOperating

    @property
    def W_designGross(self):
        """Design gross weight at the given flight conditions/loads, in kg."""
        summation = np.nansum((
            self.W_basicEmpty,
            self.usefulLoadGroup.nansum()
        ))
        return summation

    W_AUW = W_designGross

    @property
    def W_zeroFuel(self):
        """Zero fuel weight, aircraft weight without disposable fuel, in kg."""
        summation = np.nansum((
            self.W_designGross,
            -1.0 * self.usefulLoadGroup.fuel_usable
        ))
        return summation

    W_ZFW = W_zeroFuel
