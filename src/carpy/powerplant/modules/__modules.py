from carpy.powerplant.io import IOBus, IOType

__all__ = ["PlantModule"]
__author__ = "Yaseen Reza"


class PlantModule:

    def __init__(
            self, in_types: tuple[IOType.AbstractPower.__class__] | IOType.AbstractPower.__class__ = None,
            out_types: tuple[IOType.AbstractPower.__class__] | IOType.AbstractPower.__class__ = None
    ):
        # Recast as tuples
        if in_types is not None and not isinstance(in_types, tuple):
            in_types = (in_types,)
        if out_types is not None and not isinstance(out_types, tuple):
            out_types = (out_types,)

        # Pass tuples of permitted IO types to IO bus instance
        self._inputs = IOBus(*in_types) if in_types is not None else IOBus()
        self._outputs = IOBus(*out_types) if out_types is not None else IOBus()

    def __repr__(self):
        repr_str = f"<{PlantModule.__name__} '{type(self).__name__}' @ {hex(id(self))}>"
        return repr_str

    @property
    def inputs(self) -> IOBus:
        """Inputs to this plant module."""
        return self._inputs

    @property
    def outputs(self) -> IOBus:
        """Outputs from this plant module."""
        return self._outputs

    def __ilshift__(self, other):
        """Use to indicate that other plant module feeds self."""
        return other.__irshift__(self)

    def __irshift__(self, other):
        """Use to indicate that own plant module feeds other."""
        if isinstance(other, cls := PlantModule):
            assert other.inputs in self.outputs, f"{other} cannot establish I/O with {self}"

            # Record the connection
            self.outputs.add(other)
            other.inputs.add(self)

        # Elif other is a well-defined power magnitude, assert that its addition would be legal
        elif isinstance(other, IOType.AbstractPower):
            error_msg = f"'{cls.__name__}' does not output '{type(other).__name__}' power type"
            assert type(other) in self.outputs.legal_types, error_msg

            # Record the connection
            self.outputs.add(other)

        else:
            error_msg = f"{other} was deemed not to be of either type {cls.__name__} or {IOType.AbstractPower.__name__}"
            raise TypeError(error_msg)

        return self

    def __ixor__(self, other):
        """Use to indicate bidirectional connection between own and other plant module."""
        self.__ilshift__(other)
        self.__irshift__(other)
        return self
