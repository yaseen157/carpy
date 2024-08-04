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

<img style="float: right; padding-left:20px;" src="docs/source/_static/carpy.svg" width="180" height="160"/>
<h1 align="center"><p>Consolidated Aircraft Recipes in Python</p></h1>
<h4 align="center">by: Yaseen Reza</h4>

---



[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.10_--_3.12-blue.svg)](https://www.python.org/downloads/)
[![PyPI Version](https://badge.fury.io/py/carpy.svg)](https://badge.fury.io/py/carpy)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/yaseen157/carpy/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/yaseen157/carpy/tree/main)
![Jupyter Badge](https://img.shields.io/badge/jupyter-notebook-orange.svg)

CARPy is an open source project for those interested in the methodology and
approach to the conceptual-level design of fixed-wing aircraft. This library
provides its users with access to a variety of design tools for conceptual
analysis - CARPy is designed to complement, not substitute, the comprehensive
and detailed study of vehicle concepts.

- [Source Code](https://github.com/yaseen157/carpy)
- [Jupyter Notebooks](https://github.com/yaseen157/carpy/tree/main/docs/source)
- [Contributors Guide](CONTRIBUTORS_GUIDE.md)

Users will find:

- Virtual (design) atmospheres
- Hassle-free conversions between systems of units
- `[WorkInProgress]` Constraint Analysis Methods
- `[WorkInProgress]` Propulsion models and Engine Performance Decks
- *...and much more*

CARPy is continually tested to ensure you have the most stable code. Users encountering any issues should raise a GitHub
issue, or better yet, consider contributing to the project.\
**Continuous Integration Status** (CircleCI):\
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/yaseen157/carpy/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/yaseen157/carpy/tree/main)

For a detailed description of the library, please consult the documentation. To
get started, follow the instructions below.

## ✔️ Getting Started

### Installation

`carpy` is written for (and tested in) Python version 3.10+.

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

### A 'Hello world' example: Atmospheric properties

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
print(f"{isa} density at 41,000 feet:", isa.density(h=Quantity(41_000, "ft")))
```

You should see the following output:

    ISO 2533:1975 Standard Atmosphere density at 41,000 feet: 0.28740209 kg m⁻³

You can learn more about `CARPy`'s capabilities through the exemplary
[notebooks](docs/source/).

## 🐍 Acknowledgements

This project would not have been possible without the careful supervision of Dr.András Sóbester and the support of my
close colleagues and friends - you know who you are!
