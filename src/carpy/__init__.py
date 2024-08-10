"""
Consolidated aircraft recipes in Python (carpy)

This software package contains a variety of tools for the conceptual analysis
of aircraft in handy, pre-packaged modules. Check out the GitHub page for more
information, examples, and tutorials on usage.

carpy is distributed under the GNU GPLv3 License. A full copy of the license
should be present alongside the source code on GitHub.
"""
import importlib as _importlib

# The version number of the *entire* package is defined here
__version__ = "0.1.0.dev1"

# ============================================================================ #
# -------------------------------- LICENSING --------------------------------- #
# ============================================================================ #
__license__ = "GNU GPLv3"
# Console interaction copyright notice
__copyright__ = (
    f"\nCARPy  Copyright (C) 2024  Yaseen Reza"
    f"\nThis program comes with ABSOLUTELY NO WARRANTY; "
    f"for details type `CARPy.w()'."
    f"\nThis is free software, and you are welcome to redistribute it "
    f"under certain conditions; type `CARPy.c()' for details."
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

submodules = [
    "airworthiness",
    "chemistry",
    "concepts",
    "environment",
    "gaskinetics",
    "utility",
    "visual"
]

__all__ = submodules + ["__version__"]


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'carpy.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'carpy' has no attribute '{name}'")
