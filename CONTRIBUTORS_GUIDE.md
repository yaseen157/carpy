Consolidated Aircraft Recipes in Python - a Guide for Contributors
=============================================================

Thank you for considering contributing to carpy! Please follow our guidelines on
bset coding practices...

General principles
------------------

1. Follow the [PEP 8 principles](https://www.python.org/dev/peps/pep-0008/) as
   far as a possible, but never at the expense of readability or execution
   speed.

2. The consistency of physical units in the library is maintained using the
   `carpy.libutils.recastasarray` function and `carpy.tools.Quantity` objects.
    1. `recastasarray(...)`should be used to sanitise numerical inputs, ensuring
       that the following code is always dealing with `numpy.ndarray` objects.
    2. Wherever possible, the numbers returned to a user (if dimensioned) should
       be returned within a `Quantity` object. The result of adherence to this
       practice is that values will *always* be cast to their SI-base unit form.
    3. Numerical, dimensioned inputs to a function or method should *always*
       accept values corresponding with the appropriate SI-base units. The
       intended result of such behaviour is that `Quantity` objects (which cast
       their values to SI-base upon instantiation) will always pass the correct
       values. Upon recasting as array, `Quantity` objects become
       `numpy.ndarray` objects (which can then be manipulated as usual).

3. Attribute/function/method names. Parameters relating to thermodynamic
   equation of state (such as 'p' for pressure, 'T' for temperature, 'gamma' for
   ratio of specific heats) may be left in their short-hand symbolic form - so
   long as the attribute/function/method is clearly documented and indicates the
   nature of the property. While helpful, it is not necessary to document the
   units of the return value as this can be handled by the `Quantity` objects.

4. Each function/method exposed to the user must have a docstring containing
   three main sections: Parameters, Returns and a simple Example (plus an
   optional Notes section if it helps with clarity and usability, e.g., to place
   the item in a broader context). Always check afterwards that your docstring
   has rendered correctly on `readthedocs`.

5. Beyond the simple example in each docstring, a more extensive case study,
   with detailed, step-by-step explanations, should be included in a Jupyter
   notebook and placed in carpy/docs/carpy/notebooks.

6. Every new function/method needs a test, the more extensive the better. The
   test should be based on public domain data. Please include a reference to the
   data, going back as close to its original source as you can get. If your new
   code/test has new dependencies, remember to add these to
   the `requirements.txt` file at the top level of this directory structure.

7. If your code requires a unit conversion not provided by the `Quantity`
   objects, please add missing conversions to the 'quantities.xlsx' worksheet
   (along with appropriate dimensions as all the other units have). Look to
   other units in the spreadsheet for guidance on how to make a new entry - if a
   unit conversion requires a scalar multiple and transform, pay careful
   attention to the special code in `_quantities.py` that handles temperature
   converisons. Make sure to include an appropriate test for this exceptional
   case.

8. Don't Repeat Yourself - never write the same code in multiple different
   places. "Every piece of knowledge must have a single, unambiguous,
   authoritative representation within a system." (Hunt and Thomas)

Happy coding!