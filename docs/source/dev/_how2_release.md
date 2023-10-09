Releasing CARPy into the Wild
=============================

This document serves to provide an overview on building and releasing new
versions of CARPy.

## Step 1) Check what you've made

- Does this duplicate functionality already in the library?
- Is your new work in a logical or intuitive part of the library?
- Is the code itself, intuitive?
- Does your code pass unit tests?
    - Modifying existing code could affect dependents! Run *all* the tests!
    - New code should include unit tests for supported use cases
- Is it properly documented? Will an outside observer understand your work?
    - If modifying existing work, do documentation notebooks need to be re-run?

## Step 2) Check the library is ready

- Have you pulled and merged the latest version of the library from GitHub?
- Are new dependences included in `pyproject.toml` and `requirements.txt`?
- Did you update the version number in `carpy/src/carpy/__init__.py`?
- You're absolutely sure those unit tests are passing (after your latest pull)?
- Did you add new resource files? Check `pyproject.toml` knows their extensions!

## Step 3) Push changes to GitHub

- Does the library pass CI/CD tests for the Python versions we support?

1. After accepting a pull request, create a release and tag appropriately.
2. Check that you've described the release properly, and labelled if the release
   is pre-release (non-production ready) or not.

## Step 4) Build the library

See: [Official Packaging Instructions](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

1. Generate distribution archives
2. Upload distribution archives to GitHub's release/tags section.
3. If set to "pre-release", package will NOT appear on PyPI. Ensure you have the
   correct choice selected.
4. Profit
