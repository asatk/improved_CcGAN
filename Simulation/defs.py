"""
This module holds many important project definitions
used in many modules in different locations
"""

from os import getcwd
from os.path import dirname
# PROJECT_DIR = dirname(getcwd())
PROJECT_DIR = "/home/asatk/Documents/code/cern/TM"

phi_bins = 300
# phi_bins = 3
phi_min = 0.
phi_max = 3000.      #in GeV
# omega_bins = 200
# omega_bins = 2
omega_bins = phi_bins
omega_min = 0.
# omega_max = 2000./1000.    #in MeV
omega_max = phi_max