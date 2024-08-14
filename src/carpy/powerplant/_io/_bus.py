"""Module defining the IOBus, which allows power plant modules to define their input and output connection types."""
from carpy.powerplant._io import _types as IOType

__all__ = ["IOBus", "IOType"]
__author__ = "Yaseen Reza"


class IOBus(set):
    _legal_types: tuple

    def __init__(self, *args: IOType.AbstractPower):
        self._legal_types = args
        super(IOBus, self).__init__()
        return

    def __contains__(self, other):
        cls = type(self)
        assert isinstance(other, cls), f"Can only compute I/O difference for similar types (got {type(other)=})"
        return set(self.legal_types) & set(other.legal_types)

    @property
    def legal_types(self) -> tuple[IOType.AbstractPower]:
        """I/O types that this bus is permitted to host."""
        return self._legal_types
