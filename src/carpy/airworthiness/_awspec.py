from warnings import warn

__all__ = ["AWspec"]


class AWspec:
    _amendment = None
    _amendments: list[int]
    _categories: list[str]
    _category: str = None
    _type: int

    def __init__(self, __concept, /):
        self.__concept = __concept

    @property
    def _concept(self):
        """Parent vehicle concept."""
        return self.__concept

    @property
    def amendment(self) -> int:
        """Return the amendment version number of the type certification."""
        # If no amendment is specified but amendments exist, use the latest
        if self._amendment is None and self._amendments:
            return max(self._amendments)
        return self._amendment

    @amendment.setter
    def amendment(self, value):
        warnmsg = f"Amendments to type certifications isn't supported right now"
        warn(warnmsg, RuntimeWarning)

        if value in self._amendments:
            self._amendment = value
        else:
            errormsg = f"Amendment '{value}' is not recognised"
            raise ValueError(errormsg)

    @property
    def category(self) -> str:
        """Applicable category within the type certification."""
        # If no category is specified, use the default first category
        if self._category is None:
            return self._categories[0]
        return self._category

    @category.setter
    def category(self, value):
        if value in self._categories:
            self._category = value
        else:
            errormsg = f"Category '{value}' is not recognised"
            raise ValueError(errormsg)

    @property
    def type(self) -> int:
        """Immutable, the design type (14 CFR part number)."""
        return self._type
