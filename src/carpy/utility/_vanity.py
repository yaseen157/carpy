"""A module of tools to make things look prettier."""
import re

__all__ = []
__author__ = "Yaseen Reza"


# ============================================================================ #
# Subscripting and superscripting
# ---------------------------------------------------------------------------- #

class Unicodify(object):
    """Methods based on sub- and super- scripting in Unicode strings."""

    @staticmethod
    def subscript_all(text: str) -> str:
        """Using unicode, attempt to subscript anything in the input."""
        unicodemapping = {
            # unicode prefix: dict(zip(character2translate, unicode suffix))
            "0x208": dict(zip("0123456789+-=()", "0123456789ABCDEF")),
            "0x209": dict(zip("aeoxəhklmnpst", "0123456789ABCDEF"))
        }
        # Create a map from 'character2translate' to its unicode translation
        mapping = str.maketrans({
            target: chr(int(f"{prefix}{suffix}", 16))
            for prefix, submap in unicodemapping.items()
            for target, suffix in submap.items()
        })
        return text.translate(mapping)

    @staticmethod
    def superscript_all(text: str) -> str:
        """Using unicode, attempt to superscript anything in the input."""
        unicodemapping = {
            # unicode prefix: dict(zip(character2translate, unicode suffix))
            "0x00B": dict(zip("231", "239")),
            "0x207": dict(zip("0i456789+-=()n", "01456789ABCDEF")),
        }
        # Create a map from 'character2translate' to its unicode translation
        mapping = str.maketrans({
            target: chr(int(f"{prefix}{suffix}", 16))
            for prefix, submap in unicodemapping.items()
            for target, suffix in submap.items()
        })
        return text.translate(mapping)

    @classmethod
    def mathscript_safe(cls, text: str, /) -> str:
        """
        Replaces, when possible, text with unicode sub and/or superscript
        variants.

        Args:
            text: The string of text to apply the unicode transformation to.

        Returns:
            A string where superscript and subscript substitutions have been
            made.

        Examples: ::

            >>> Unicodify.mathscript_safe("g_{0} = 9.81 ms^{2}")
            # Outputs: g₀ = 9.81 ms²

        """
        # Identify text that should be made super or subscript
        re_numscript = re.compile(r"[\^_]{[^(?!{})]+?}")
        text2translate = set(re_numscript.findall(text))

        # Apply super or subscript translation
        for numscript in text2translate:
            translator = {
                "^": cls.superscript_all, "_": cls.subscript_all
            }[numscript[0]]
            # Use indexing [2:-1] to ignore ^{} and _{} patterns
            text = text.replace(numscript, translator(numscript[2:-1]))

        return text

    @classmethod
    def chemical_formula(cls, text: str, /) -> str:
        """
        Formats molecular formula with subscripted numbers of atoms.

        Args:
            text: The string of text on which to apply the Unicode transform.

        Returns:
            Chemical formula with subscripted numbers of atoms.

        Examples: ::

            >>> print(Unicodify.chemical_formula("2(C2H5)2O"))
            '2(C₂H₅)₂O'

        """
        prettystring = "".join([
            Unicodify.subscript_all(char)
            if (char.isnumeric() and i != 0) else char
            for i, char in enumerate(re.findall(r"\d+|[A-z()]+", text))
        ])
        return prettystring

    @classmethod
    def superscript_trailingnums(cls, text: str, /) -> str:
        """
        Formats trailing numerical characters in a word with superscripting.

        Args:
            text: The string of text on which to apply the Unicode transform.

        Returns:
            String with the trailing numbers superscripted.

        Examples: ::

            >>> print("Neon has the electron arrangement", end=" ")
            >>> print(Unicodify.superscript_trailingnums("1s2 2s2 2p6"))
            'Neon has the electron arrangement 1s² 2s² 2p⁶'

        """

        def match_handler(matchobj):
            """Take a match object and superscript all the text contained."""
            return cls.superscript_all(matchobj.group(0))

        return re.sub(r"\d+\s|\d+$", match_handler, text)


__all__ += [Unicodify.__name__]


# ============================================================================ #

class Pretty(object):
    """Assorted methods for making pretty outputs."""

    @staticmethod
    def temperature(T, unit: str = None):
        """
        Make a pretty temperature string given the temperature and unit.

        Args:
            T: Scalar quantity of temperature that can be cast as a float.
            unit: One of "degC", "degF", "degR", or "K" units of temperature.

        Returns:
            A string for neatly displaying the temperature.

        """
        # Prefix (?)
        prefix = "+" if T >= 0 else ""

        # Temperature string
        temperature = f"{float(T)}".rstrip("0") if float(T) != 0 else "0"
        temperature = temperature.rstrip(".")

        # Suffix
        unit = "degC" if unit is None else unit
        suffix = {
            "degC": u"\N{DEGREE SIGN}C",
            "degF": u"\N{DEGREE SIGN}F",
            "degR": u"\N{DEGREE SIGN}R",
            "K": " K"
        }[unit]

        temperature_string = f"{prefix}{temperature}{suffix}"

        return temperature_string

    @staticmethod
    def signed(text, /):
        """
        Given a string representing a number, add a plus-sign to the front of
        the string if it represents a value above 0.

        Args:
            text: A string of text representing the number to make pretty.

        Returns:
            A signed string of text.

        """
        # Hopefully try-except architecture this is input-type agnostic
        try:
            number = float(text)
        except ValueError:
            pass
        else:
            if number > 0:
                text = f"+{text}"
        finally:
            return f"{text}"


__all__ += [Pretty.__name__]
