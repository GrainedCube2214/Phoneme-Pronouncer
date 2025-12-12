#!/usr/bin/env python3
"""
One-click demo launcher for Phoneme Pronunciation Checker
Auto-installs everything needed
"""

import sys
import os
import subprocess
import platform

# Check Python version
if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
    print(f"Error: Requires Python 3.10.x (you have {sys.version}.)")
    input("Press Enter to exit...")
    sys.exit(1)

else: print(f"Python version {sys.version} detected.")

print("Setting up Phoneme Pronunciation Checker...")

# Determine Python path in virtual environment
venv_python = os.path.join('env', 'Scripts', 'python.exe') if platform.system() == 'Windows' else os.path.join('env', 'bin', 'python')

# Create venv if it doesn't exist
if not os.path.exists('env'):
    print("Creating virtual environment...")
    import venv
    venv.create('env', with_pip=True)
    
    # Install packages
    print("Installing packages (this may take a few minutes)...")
    subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL)
    subprocess.check_call([venv_python, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Setup complete!")

# Run the demo
print("\n" + "="*50)
print("STARTING PHONEME PRONUNCIATION CHECKER")
print("="*50 + "\n")

try:
    subprocess.call([venv_python, "-c", "from integrated_checker import main; main()"])
except KeyboardInterrupt:
    print("\n\nGoodbye!")