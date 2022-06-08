"""
This module holds definitions of constants/hyperparameters used to define the
geometry and data of the problem being interpolated.
"""

# Use-case Parameters
phi_bins = 300
phi_min = 0.
phi_max = 3000.      #in GeV
omega_bins = 200
omega_min = 0.
omega_max = 2000./1000.    #in MeV

# Gaussian Parameters
xmin = 0.
xmax = 1.
xbins = 100

ymin = 0.
ymax = 1.
ybins = 100

xcov_change_linear_max_factor = 16.   #max change in growth in x
ycov_change_linear_max_factor = 16.   #max change in growth in y
#cov_change_skew_rate =             #radian angle rotation