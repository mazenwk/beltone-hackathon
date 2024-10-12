import subprocess
import sys
def install_requirements(requirements_file='requirements.txt'):
 try:
    # Use subprocess to run the pip install command
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
    print(f"Successfully installed packages from {requirements_file}")
 except subprocess.CalledProcessError as e:
    print(f"Failed to install packages from {requirements_file}. Error: {e}")