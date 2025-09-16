#!/usr/bin/env python3
"""
Install required R packages for the analysis
"""

import os
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, isinstalled

# Set R environment variables
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
os.environ['DYLD_LIBRARY_PATH'] = f"{os.environ['R_HOME']}/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}"

# Get R utils for package installation
utils = importr('utils')

# List of required packages
required_packages = ['arrow', 'fixest']

# Function to install packages
def install_r_packages(packages):
    """Install R packages if not already installed"""
    for package in packages:
        if not isinstalled(package):
            print(f"Installing R package: {package}...")
            utils.install_packages(package, repos='https://cran.rstudio.com/')
            print(f"✅ {package} installed successfully")
        else:
            print(f"✅ {package} is already installed")

# Install packages
print("Checking and installing required R packages...")
print("-" * 40)
install_r_packages(required_packages)
print("-" * 40)
print("All required R packages are ready!")