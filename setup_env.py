import subprocess
import os

# Define packages to install
pkgs = [
    "opencv-python",
    "ultralytics",
    "numpy",
    "python-dotenv"
]

# Path to the virtual environment's Python interpreter
venv_python = os.path.join(".venv", "bin", "python")

# Ensure virtual environment exists (optional, but good practice)
if not os.path.exists(os.path.join(".venv", "pyvenv.cfg")):
    print("Creating virtual environment...")
    subprocess.run([venv_python, "-m", "venv", ".venv"], check=True)

# Install packages using the venv's pip
print(f"Installing packages: {', '.join(pkgs)}...")
install_command = [venv_python, "-m", "pip", "install", "--break-system-packages", "-q"] + pkgs

try:
    result = subprocess.run(install_command, check=True, capture_output=True, text=True)
    print("Installation successful!")
    # print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Installation failed!")
    print(e.stderr)
except FileNotFoundError:
    print(f"Error: Virtual environment Python interpreter not found at {venv_python}.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Verify installation
print("Verifying installation...")
verify_command = [venv_python, "-m", "pip", "show", "opencv-python", "ultralytics", "numpy", "python-dotenv"]
try:
    result = subprocess.run(verify_command, check=True, capture_output=True, text=True)
    print("Verification successful! Packages found:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("Verification failed. Some packages might not be installed correctly.")
    print(e.stderr)
