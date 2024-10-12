import os
import subprocess
import sys

# Create Libraries folder if it doesn't exist
libraries_dir = 'Libraries'
if not os.path.exists(libraries_dir):
    os.makedirs(libraries_dir)

# Function to download a library to the Libraries folder
def download_library(library_name):
    try:
        # Use pip to download the library into the Libraries folder
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--target', libraries_dir, library_name])
        print(f"Successfully downloaded {library_name} to {libraries_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {library_name}: {e}")
