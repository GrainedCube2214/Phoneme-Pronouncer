# verify python version == 3.10.x
import sys
if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
    raise RuntimeError("This script requires Python 3.10.x")
else:
    print("Python version is 3.10.x, proceeding...")



skip = True # Set to False to create virtual environment and install packages, True to skip (debugging purpose)
if skip:
    print("Skipping virtual environment creation and package installation. This should not happen in normal usage. Please set skip = False.")
else:
    # Create a virtual environment
    import venv
    venv.create('env', with_pip=True)
    print("Virtual environment 'env' created successfully.")

    # download and install required packages
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])



print("Starting the demo...")

from integrated_checker import main
main()