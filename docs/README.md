<h1 align="center">
    <img style="float: right; padding-bottom:40px;" src="https://raw.githubusercontent.com/yaseen157/carpy/main/branding/logo_primary.png" width="300"/>
</h1>

# CARPy Documentation

Users of the library can find the following documentation in this directory:

- `/source` contains all docs and assets.
    - `/demo` contains examples demonstrating the library.
    - `/developer` contains guides for developers.
    - `/tutorial` contains step-by-strp user guides and detailed walkthroughs.

> ⚠️ **IMPORTANT:**
> _For users encountering `ModuleNotFoundError` when running the Jupyter notebooks_
>
> If you cloned this project and made an editable installation of the source code, the iPython backend of Jupyter
> notebooks may have trouble locating CARPy.
> Usually, this is solved by running the following code at the start of your notebook, being sure to change the
> directory to point to your CARPy `/src` folder:

```python
import sys

# The code for carpy is located inside the carpy project's "src" folder
src_loc = r"C:\Users\...\carpy\src"
sys.path.insert(0, src_loc)
```

> Instead of doing this yourself in every affected notebook, you can optionally trust CARPy to tell the `iPython`
> interpreter to run an equivalent script everytime it starts up.
> Have a go at running the `ipynb_wizard.py` file that can be located in the folder - it works by locating your system's
> default iPython profile and depositing a startup script that prepends the `sys.path` variable with CARPy's location. 