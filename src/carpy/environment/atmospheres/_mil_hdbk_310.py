"""
Module implementing the U.S. Department of Defence Handbook 310's atmosphere profiles.

References:
    MIL-HDBK-310 Global Climatic Data for Developing Military Products

"""
import os
import re

import pandas as pd

from carpy.environment.atmospheres import StaticAtmosphereModel
from carpy.utility import PathAnchor

__all__ = []
__author__ = "Yaseen Reza"


def parse_atmprofile_notation(x) -> float:
    """Interpret MIL-HDBK-310 atmospheric profile notation for floats as scientific notation."""
    if isinstance(x, (int, float)):
        return x
    unspaced_x = x.replace(" ", "")
    pattern = r"([\d.]+)(?:([+-])([^*]+)\*?)?"  # 253.1 --> (253.1, ,); 1.203+0* --> (1.203, +, 0)
    replace = r"\1e\g<2>0\3"  # (1.203, +, 0) --> 1.203e+00
    float_value = eval(re.sub(pattern, replace, unspaced_x))
    return float_value


# Load the handbook data
anchor = PathAnchor()
filename = "MIL-HDBK-310_profiles.xlsx"
filepath = os.path.join(anchor.directory_path, "..", "data", filename)

# Maybe it's better to load the handbook data dynamically, when its needed?
dataframes: dict = pd.read_excel(
    filepath, sheet_name=None,
    converters={i: parse_atmprofile_notation for i in range(0, 5)}
)


# Define a class factory
def make_milhdbk310_class(key: str):
    """Spawn a class that will *hopefully* stay in the runtime environment once spawned."""
    if key not in dataframes:
        error_msg = f"Could not find {key=} in available data. The available keys are: {dataframes.keys()}"
        raise ValueError(error_msg)

    return type(f"MIL_HDBK_310_{key}", (StaticAtmosphereModel,), {})


raise NotImplementedError("This file is a work in progress. Keep it out of your imports!")
