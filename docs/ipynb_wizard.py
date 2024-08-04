"""Walk a user through setting up their iPython kernel to locate CARPy."""
import os
import re
import subprocess
import time
import warnings

__all__ = []
__author__ = "Yaseen Reza"


def run_wizard(target_profile_name: str = None, sleep: bool = True):
    """
    Run a setup wizard that places the CARPy library into the namespace of Jupyter notebook iPython kernels.

    The wizard locates a profile for the iPython kernel, and assumes that the Jupyter notebook kernel will use this
    profile. In here, it conjures a file that adds CARPy to the system path.

    Args:
        target_profile_name: The name of the iPython configuration profile to modify. Optional, defaults to 'default'.
        sleep: Whether the wizard sleeps for three seconds upon code completion. Its purpose is to ensure a user can at
            least be informed of code success. Optional, defaults to True.

    """
    print(f"Running CARPy to iPython import wizard...")
    default_profile = "default"

    # Find the iPython configuration path
    result = subprocess.run(["ipython", "locate"], stdout=subprocess.PIPE)
    ipython_path = result.stdout.decode("utf-8").strip()

    # Determine the available selection of profiles
    def find_ipython_profiles() -> set[str]:
        """Locate available iPython profiles in current working directory."""
        regex = re.compile("profile_(.+)")
        return {regex.match(x).groups()[0] for x in os.listdir(ipython_path) if regex.match(x)}

    # For basic users, the Jupyter notebook is using the default_profile. If an advanced user had setup something else,
    #   they probably didn't need the help of the wizard anyway...
    if target_profile_name is None:

        if default_profile not in find_ipython_profiles():
            warn_msg = f"Default iPython profile '{default_profile}' was not discovered at: {ipython_path}"
            warnings.warn(message=warn_msg, category=RuntimeWarning)
            print("Creating iPython profile...")
            print(subprocess.run(["ipython", "profile", "create"], stdout=subprocess.PIPE))
            assert default_profile in find_ipython_profiles(), "Subprocess was unable to create default iPython profile"

        target_profile_name = default_profile

    elif target_profile_name not in (profiles := find_ipython_profiles()):
        error_msg = f"{target_profile_name=} was not located among available {profiles=}"
        raise RuntimeError(error_msg)

    # Locate the startup folder in the profile
    startup_path = os.path.join(ipython_path, f"profile_{target_profile_name}", "startup")
    filename = "carpy_source.py"
    filepath = os.path.join(startup_path, filename)

    # Try first to look for a globally accessible install of CARPy. If unavailable, try to locate one with relative path
    try:
        import carpy
        absolute_src = os.path.abspath(os.path.join(carpy.__path__[0], "../"))
    except ModuleNotFoundError:
        warn_msg = f"Module 'carpy' does not appear to be installed, checking local namespace instead"
        warnings.warn(message=warn_msg, category=RuntimeWarning)
        absolute_src = os.path.abspath(os.path.join(__file__, "../../src"))
        assert "carpy" in os.listdir(absolute_src), f"Couldn't locate the 'carpy' library at {absolute_src}"

    # Check that it's safe to write the iPython - CARPy linking file
    if filename in os.listdir(startup_path):

        # If the file exists, find the CARPy source that it's pointing to
        with open(filepath, "r") as f:
            regex = re.compile("[\"'](.+)[\"']")
            existing_srcs = tuple(filter(lambda x: x.startswith("src_loc"), f.read().splitlines()))
            existing_srcs = {regex.search(x).groups()[0] for x in existing_srcs if regex.search(x)}

        # And make sure it is safe to delete (it's pointing the same place we were going to point anyway)
        if len(existing_srcs) == 0:
            error_msg = (
                f"A copy of '{filename}' already exists at {startup_path} and points to a different install of CARPy. "
                f"Make a backup of that file, delete it, or temporarily rename it to deconflict with the wizard"
            )
            raise FileExistsError(error_msg)
        if existing_srcs - {absolute_src}:
            error_msg = (
                f"A copy of '{filename}' already exists at {startup_path} and points to a different install of CARPy. "
                f"Make a backup of that file, delete it, or deconflict your multiple installs of the CARPy library"
            )
            raise FileExistsError(error_msg)

    # It is now safe to write the file
    print(f"{absolute_src=}")
    print(f"Writing to '{filename}' at path: {startup_path}")
    with open(filepath, "w") as f:
        f.writelines([
            f'"""Start-up file to help iPython find CARPy notebooks."""\n',
            f'import sys\n\n'
            f'src_loc = r"{absolute_src}"\n'
            f'sys.path.insert(0, src_loc)\n'
        ])

    print("Done!", end=" ")
    if sleep:
        print("(sleeping for 3 seconds)", end="")
        time.sleep(3)
    print("")
    return


if __name__ == "__main__":
    run_wizard()
