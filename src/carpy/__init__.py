"""
Consolidated aircraft recipes in Python (carpy)

This software package contains a variety of tools for the conceptual analysis
of aircraft in handy, pre-packaged modules. Check out the GitHub page for more
information, examples, and tutorials on usage.

carpy is distributed under the GNU GPLv3 License. A full copy of the license
should be present alongside the source code on GitHub.
"""
# The version number of the "entire" package is specified here
__version__ = "0.0.6-3"

# ============================================================================ #
# -------------------------------- LICENSING --------------------------------- #
# ============================================================================ #
__license__ = "GNU GPLv3"
# Console interaction copyright notice
__copyright__ = (
    f"\nADRpy  Copyright (C) 2023  Yaseen Reza"
    f"\nThis program comes with ABSOLUTELY NO WARRANTY; "
    f"for details type `ADRpy.w()'."
    f"\nThis is free software, and you are welcome to redistribute it "
    f"under certain conditions; type `ADRpy.c()' for details."
)


def c() -> None:
    """Redistribution information"""
    disttext = (
        "This program is free software: you can redistribute it and/or modify\n"
        "it under the terms of the GNU General Public License as published by\n"
        "the Free Software Foundation, either version 3 of the License, or\n"
        "(at your option) any later version."
    )
    print(disttext)
    return None


def w() -> None:
    """Warranty information"""
    warrantytxt = (
        "This program is distributed in the hope that it will be useful,\n"
        "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
        "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
        "GNU General Public License for more details."
    )
    print(warrantytxt)
    return None

# ============================================================================ #
