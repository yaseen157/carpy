"""Base class definition for component 'modules' of a larger powerplant (that networks such modules together)."""
import typing
# import warnings

import numpy as np

from carpy.powerplant import IOBus, IOType

__all__ = ["PlantModule"]
__author__ = "Yaseen Reza"


class PlantModule:
    """
    Base class for modules of the power plant network.

    Methods:
        inputs: Instance property. Describes the set of other power plant modules that are feeding this module.
        outputs: Instance property. Describes the set of other power plant modules that this module feeds.
        admit_low: Instance property. A parameter that defines the lower bound for a module's generic control.

    """
    _inputs: IOBus
    _outputs: IOBus
    _admittance = 1.0
    _admit_low = 1.0

    def __init__(
            self, name: str = None,
            in_types: tuple[IOType.AbstractFlow.__class__] | IOType.AbstractFlow.__class__ = None,
            out_types: tuple[IOType.AbstractFlow.__class__] | IOType.AbstractFlow.__class__ = None
    ):
        """
        Args:
            in_types: IOType objects that describe the valid types of the power plant module's inputs.
            out_types: IOType objects that describe the valid types of the power plant module's outputs.
        """
        self.name = name

        # Recast as tuples
        if in_types is not None and not isinstance(in_types, tuple):
            in_types = (in_types,)
        if out_types is not None and not isinstance(out_types, tuple):
            out_types = (out_types,)

        # Pass tuples of permitted IO types to IO bus instance
        self._inputs = IOBus(*in_types) if in_types is not None else IOBus()
        self._outputs = IOBus(*out_types) if out_types is not None else IOBus()

    def __repr__(self):
        if self.name is None:
            repr_str = f"<{PlantModule.__name__} '{type(self).__name__}' @ {hex(id(self))}>"
        else:
            repr_str = f"<{PlantModule.__name__} '{self.name}'>"
        return repr_str

    @property
    def inputs(self) -> IOBus:
        """Fixed inputs to this plant module."""
        return self._inputs

    @property
    def outputs(self) -> IOBus:
        """Fixed outputs from this plant module."""
        return self._outputs

    def __ilshift__(self, other):
        """Use to indicate that other plant module feeds self."""

        # If other turns out to be a well-defined power magnitude, assert that its addition would be legal
        if isinstance(other, IOType.AbstractFlow):
            error_msg = f"'{self}' does not allow input from '{type(other).__name__}' power type"
            assert type(other) in self.inputs.legal_types, error_msg

            # Record the connection
            self.inputs.add(other)

        else:
            other.__irshift__(self)  # Carry out reversed __irshift__ to record connection of self and other
        return self  # ... but remember to return self, so the "in-place" part if ilshift still makes sense

    def __irshift__(self, other):
        """Use to indicate that own plant module feeds other."""
        if isinstance(other, cls := PlantModule):
            assert other.inputs in self.outputs, f"{other} cannot establish I/O with {self}"

            # Record the connection
            self.outputs.add(other)
            other.inputs.add(self)

        # Elif other is a well-defined power magnitude, assert that its addition would be legal
        elif isinstance(other, IOType.AbstractFlow):
            error_msg = f"'{self}' does not allow output to '{type(other).__name__}' power type"
            assert type(other) in self.outputs.legal_types, error_msg

            # Record the connection
            self.outputs.add(other)

        else:
            error_msg = f"{other} was deemed not to be of either type {cls.__name__} or {IOType.AbstractFlow.__name__}"
            raise TypeError(error_msg)

        return self

    def forward(self, *inputs):
        """Template for plant module power propagation."""
        error_msg = f"{PlantModule.__name__} object '{type(self).__name__}' does not implement the 'forward' method"
        raise NotImplementedError(error_msg)
