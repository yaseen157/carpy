"""Module defining the IOBus, which allows power plant modules to define their input and output connection types."""
from carpy.powerplant._io_type import IOType

__all__ = ["IOBus"]
__author__ = "Yaseen Reza"


class IOBus(set):
    """
    A subclass of set used to explicitly gate the inputs and outputs to power plant modules.

    When instantiated with arguments, an object of this class uses these arguments to determine what kinds of I/O are
    legal for this input/output bus. Any subsequent attempts to add I/O objects to this set are subjected to checks to
    make sure the added object is one of the 'legal types' of object.
    """
    _legal_types: tuple

    def __init__(self, *args: IOType.AbstractPower.__class__):
        self._legal_types = args
        super(IOBus, self).__init__()
        return

    def __contains__(self, other):
        cls = type(self)
        assert isinstance(other, cls), f"Can only compute I/O difference for similar types (got {type(other)=})"
        return set(self.legal_types) & set(other.legal_types)

    @property
    def legal_types(self) -> tuple[IOType.AbstractPower.__class__]:
        """I/O types that this bus is permitted to host."""
        return self._legal_types

    def add(self, __element):
        error_msg = f"{__element} is not a permitted IO type (see allowed types in the '{self}.legal_types' attribute)"
        assert isinstance(__element, self.legal_types), error_msg
        super().add(__element)
