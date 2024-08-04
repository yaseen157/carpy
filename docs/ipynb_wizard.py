"""Walk a user through setting up their iPython kernel to locate CARPy."""
import os
import subprocess

__all__ = []
__author__ = "Yaseen Reza"


def run_wizard():
    """Walk a user through setting up their iPython configuration and enable access to the carpy package."""

    # Find where iPython keeps its configuration
    result = subprocess.run(["ipython", "locate"], stdout=subprocess.PIPE)
    ipython_path = result.stdout.decode("utf-8").strip()

    # Locate available iPython profiles
    def find_ipython_profiles() -> set[str]:
        """Locate available iPython profiles in current working directory."""
        return {x for x in os.listdir(ipython_path) if x.startswith("profile_")}

    # Choose a profile to modify
    selected_profile = None
    if len(profiles := find_ipython_profiles()) == 0:

        # Create a profile
        while True:
            print(f"No iPython profiles were detected at {ipython_path}")
            print(f"Do you want to create one? (y/n)")
            choice = input(">>> ").lower()

            if choice == "n":
                print("No profile was created. Quitting")
                quit()
            elif choice == "y":
                print("Creating profile...")
                result = subprocess.run(["ipython", "profile", "create"], stdout=subprocess.PIPE)
                print(result)
                break
            else:
                print(f"Unrecognised choice '{choice}'")

        # If we succeeded in making a profile, select the profile
        profiles = find_ipython_profiles()
        assert len(profiles) > 0, f"Could not create an iPython profile at {ipython_path}"
        selected_profile = profiles.pop()

    else:

        # Choose an existing profile
        while selected_profile is None:
            print("Which iPython profile should be modified?")
            print(f"Select one of: {profiles}, or type 'quit()' to leave the wizard")
            choice = input(">>> ")

            if choice in profiles:
                selected_profile = choice
                break
            elif choice.lower() == "quit()":
                exit()
            else:
                print(f"Unrecognised choice '{choice}'")

    # Check that it's safe to proceed
    startup_path = os.path.join(ipython_path, selected_profile, "startup")
    filename = "carpy_source.py"
    if filename in os.listdir(startup_path):
        while True:
            print(f"The file '{filename}' already exists in {startup_path}.")
            print(f"Do you want to overwrite the contents of {filename}? (y/n)")
            choice = input(">>> ")

            if choice == "n":
                print("Consider making a backup of the file before proceeding further")
                exit()
            elif choice.lower() == "y":
                break
            else:
                print(f"Unrecognised choice '{choice}'")

    # Locate CARPy's source code (the target)
    try:
        import carpy
        target = os.path.abspath(os.path.join(carpy.__path__[0], "../"))  # absolute location, if possible
    except ModuleNotFoundError:
        target = os.path.abspath(os.path.join(os.getcwd(), "../../src"))  # relative location otherwise!
        assert "carpy" in os.listdir(target), f"Couldn't locate the 'carpy' library at {target}"
    print(f"{target=}")

    # ...proceed with creating the configuration
    filepath = os.path.join(startup_path, filename)
    print(f"Creating '{filename}' at path: {startup_path}")
    with open(filepath, "w") as f:
        f.writelines([
            f'"""Start-up file to help iPython find CARPy notebooks."""\n',
            f'import sys\n\n'
            f'src_loc = r"{target}"\n'
            f'sys.path.insert(0, src_loc)\n'
        ])

    print("Done!")


if __name__ == "__main__":
    run_wizard()
