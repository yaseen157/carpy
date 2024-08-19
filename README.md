<!--
    Consolidated Aircraft Recipes in Python (carpy)
    Copyright (C) 2024  Yaseen Reza

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

<h1 align="center">
<img src="https://raw.githubusercontent.com/yaseen157/carpy/main/branding/logo_primary.png" width="300">
</h1><br>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.10_--_3.12-blue.svg)](https://www.python.org/downloads/)
[![PyPI Version](https://badge.fury.io/py/carpy.svg)](https://badge.fury.io/py/carpy)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/yaseen157/carpy/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/yaseen157/carpy/tree/main)
![Jupyter Badge](https://img.shields.io/badge/jupyter-notebook-orange.svg)

CARPy is an open source project for those interested in the methodology and
approach to the conceptual-level design of fixed-wing and rotor-wing aircraft. This library
provides its users with access to a variety of design tools for conceptual
analysis. The full terms of software use should have been
provided to you with a copy of the software, i.e. the work here-in is protected by a standard GPLv3 license.

Users will find:

- Virtual (design) atmospheres and environments
- Hassle-free conversions between systems of units
- `[WorkInProgress]` Constraint Analysis Methods
- `[WorkInProgress]` Propulsion models and Engine Performance Decks
- *...and much more*

CARPy is continually tested to ensure you have the most stable code. Users encountering any issues should raise a GitHub
issue, or better yet, consider contributing to the project.

- [Source Code](https://github.com/yaseen157/carpy)
- [Jupyter Notebooks](https://github.com/yaseen157/carpy/tree/main/docs/source)
- [Contributors Guide](CONTRIBUTORS_GUIDE.md)

**Continuous Integration Status** (CircleCI):\
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/yaseen157/carpy/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/yaseen157/carpy/tree/main)

For a detailed description of the library, please consult the documentation. To
get started, follow the instructions below.

# ✔️ Getting Started

## Installation

CARPy is written for (and tested in) Python version 3.10+.

### Method 1: Using the Python Package Index

![Terminal](https://badgen.net/badge/icon/terminal?icon=terminal&label)
[![PyPi](https://badgen.net/badge/icon/pypi?icon=pypi&label)](https://https://pypi.org/)

On most systems you should be able to simply open an operating system terminal
and at the command prompt type

    $ pip install carpy

or

    $ python -m pip install carpy

NOTE: `pip` is a Python package; if it is not available on your system, download
[get-pip.py](https://bootstrap.pypa.io/get-pip.py) and run it in Python by
entering

    $ python get-pip.py

at the operating system prompt.

If you already have a version of carpy installed and are simply trying to
upgrade, use the `--upgrade` flag:

    $ pip install --upgrade carpy

### Method 2: Using git clone

![Terminal](https://badgen.net/badge/icon/terminal?icon=terminal&label)
[![git](https://badgen.net/badge/icon/git?icon=git&label)](https://git-scm.com)

An alternative approach to installing carpy is to clone the GitHub repository
using `git`, by typing

    $ git clone https://github.com/yaseen157/carpy.git

at the command prompt. Following a successful clone of files to your machine,
navigate to the library root (this contains the file `pyproject.toml`). At this
point, you may enter the following:

    $ python -m pip install ./

Alternatively, adventurous users who want an editable install to make any
customisations in their local build should use the `--editable` flag:

    $ python -m pip install -e ./

Should you find that your installation requires packages you do not have in your
current Python environment, install them by typing this in the same prompt:

    $ python -m pip install -r requirements.txt

## 'Hello world' examples

### Atmospheric properties

There are several options for running the examples shown here: you could copy
and paste them into a `.py` file, save it and run it in Python, or you could
enter the lines, in sequence, at the prompt of a Python terminal. You could also
copy and paste them into a Jupyter notebook
(`.ipynb` file) cell and execute the cell.

```python
"""'Hello World' example to introduce users to CARPy atmospheres."""
from carpy.environment.atmospheres import ISA
from carpy.utility import Quantity

# Instantiate an atmosphere model
isa = ISA()

# Query the ambient density in this model at 41,000 feet
print(f"{isa} density at 41,000 feet:", isa.density(z=Quantity(41_000, "ft")))
```

You should see the following output:

    ISO 2533:1975 Standard Atmosphere density at 41,000 feet: 0.28740209 kg m⁻³

You can learn more about `CARPy`'s capabilities through the exemplary
[notebooks](https://github.com/yaseen157/carpy/tree/main/docs/source).

# 🤔 Getting Help

CARPy was developed explicitly to fill a gap in the market for a well documented, open-source, and highly modular suite
of tools for analysing aircraft concepts.

If you feel that our documentation leaves something to be desired or want to get in touch about any bugs you've
discovered, let us know about it by raising a GitHib issue.

## Uninstalling with PIP

![Terminal](https://badgen.net/badge/icon/terminal?icon=terminal&label)
[![PyPi](https://badgen.net/badge/icon/pypi?icon=pypi&label)](https://https://pypi.org/)

If CARPy did not meet your expectations or you are looking to reinstall the library, you can use `pip` to remove CARPy
from your system.

Open a command prompt anywhere in your CARPy enabled
Python environment. You don't need to navigate to a specific folder to uninstall
CARPy, as pip already knows where CARPy lives on your machine. Type as follows:

    $ pip uninstall -y carpy

Which tells pip to uninstall any package on your machine it knows to be called
CARPy, and uses the optional flag `-y` to answer "yes" automatically to any
prompt asking the user if they want to uninstall.

> 📝 **Note:** It's not uncommon for Python users to make use of "virtual
> environments." These behave like isolated installations of Python, so for scientific
> or development purposes you can be sure your code depends on exactly the
> files and libraries you want it to. Make sure you're in the correct
> environment when you're uninstalling, or nothing will happen. You can tell
> which environment has CARPy in because you can type in the terminal:
>
>       $ pip show carpy
> and see CARPy library info (as well as the installed version number).

# 🐍 Acknowledgements

This library was written and published by **Yaseen Reza**, in support of my studies towards the degree of doctor of
philosophy.

This project would not have been possible without the careful supervision of Dr.András Sóbester and the support of my
close colleagues and friends - you know who you are!
