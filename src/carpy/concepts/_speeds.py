class Airspeed:

    # TODO: Needs to access atmosphere objects automagically, using Quantity
    def __init__(self, altitude=None, /):
        self._altitude = altitude


class VspeedsFW:
    def __init__(self, __concept, /):
        self.__concept = __concept

    @property
    def _concept(self):
        """Parent vehicle concept."""
        return self.__concept

    @property
    def V1(self):
        """Take-off decision speed, V1."""
        raise NotImplementedError

    @property
    def V2(self):
        """Initial climb-out speed, V1."""
        raise NotImplementedError

    @property
    def VA(self):
        """Design manoeuvring speed, VA."""
        raise NotImplementedError

    @property
    def VB(self):
        """Design speed for maximum gust intensity, VB."""
        raise NotImplementedError

    @property
    def VC(self):
        """Design cruising speed, VC."""
        raise NotImplementedError

    @property
    def VD(self):
        """Design dive speed, VD."""
        raise NotImplementedError

    @property
    def VEF(self):
        """Engine Failure speed, VEF."""
        raise NotImplementedError

    @property
    def VF(self):
        raise NotImplementedError

    @property
    def VFE(self):
        raise NotImplementedError

    @property
    def VH(self):
        """Maximum level-flight speed, VH."""
        raise NotImplementedError

    @property
    def VMC(self):
        """Minimum control speed, VMC."""
        raise NotImplementedError

    @property
    def VMCG(self):
        """Minimum control speed on the ground, VMCG."""
        raise NotImplementedError

    @property
    def VMO(self):
        """Maximum operating speed, VMO."""
        raise NotImplementedError

    @property
    def VNE(self):
        raise NotImplementedError

    @property
    def VR(self):
        """
        Rotation speed, VR.

        The speed at which the pilot makes a control input with the intention of
        lifting the aeroplane out of contact with the runway or water surface.
        """
        raise NotImplementedError

    @property
    def VREF(self):
        """Reference landing approach speed, VREF."""
        raise NotImplementedError

    @property
    def VS(self):
        """Computed stalling speed with flaps retracted at design weight, VS."""
        raise NotImplementedError

    @property
    def VSF(self):
        """Computed stalling speed with flaps extended at design weight, VSF."""
        raise NotImplementedError

    @property
    def VS0(self):
        """Minimum steady flight speed in the landing configuration."""
        # TODO: Needs airspeed objects
        raise NotImplementedError

    @property
    def VS1(self):
        """Steady flight speed in a particular configuration."""
        raise NotImplementedError
