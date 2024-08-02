"""Module implementing the U.S. Department of Defence Handbook 310's atmosphere profiles."""
import os
import re

import pandas as pd

from carpy.environment.atmosphere import StaticAtmosphereModel
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
filepath = os.path.join(anchor.directory_path, "data", filename)
dataframes:dict = pd.read_excel(filepath, sheet_name=None, converters={i: parse_atmprofile_notation for i in range(0, 5)})
