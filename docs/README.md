<img style="float: right; padding-left:20px;" src="source/_static/carpy.svg" width="180" height="160"/>

# CARPy Documentation

---

Users of the library can find the following documentation in this directory:

- `source` contains library docs and assets.
    - `demo` contains examples of the library's capabilities.
    - `developer` contains guides and details for contributors and developers.
    - `tutorial` contains user guides and detailed walkthroughs.

> [!IMPORTANT]
> _For users encountering `ModuleNotFoundError` when running the Jupyter notebooks_
>
> If you have cloned this project and are making an editable install of the contained library, the iPython backend of
> Jupyter notebooks may not be able to locate CARPy. If this is the case, have a go at running the `ipynb_wizard.py`
> file. It configures the iPython kernel to effectively run the following code at the start of each notebook - and of
> course, you are welcome to ignore the wizard and use this template yourself:
> ```python
> import sys
> 
> # The code for carpy is located inside the carpy project's "src" folder
> src_loc = r"C:\Users\...\carpy\src"
> sys.path.insert(0, src_loc)
> ```