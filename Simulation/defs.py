"""
This module holds definitions of (hyper)parameters used to define the
geometry and data of the problem being interpolated.
"""

### --- Network/Run parameters --- ###
root_path = "/home/asatk/Documents/code/cern/asatk-improved_CcGAN/Simulation"   #absolute path to working directory containing project details
seed = 20               #seed for RNG
nsim = 1                #num simulations in a run
niter = 5000            #num iterations for which a simulation runs
niter_resume = 0        #num iterations to resume training on a trained GAN
niter_save = 1000       #num iterations to save GAN state
nbatch_d = 1000         #num labels in dis batch
nbatch_g = 1000         #num labels in gen batch
sigma_kernel = -1.0     #defines width of kernel/noise for training
kappa = -2.0            #vicinity parameter (for both hard and soft)
dim_gan = 2             #dimensions of the GAN, i.e., 2 for 2-D problem - also referred to as latent dimension of GAN? understand what that means
lr = 5e-5               #learning rate: keep btwn [1e-3, 1e-6]
geo = 'line'            #determines geometry/problem
thresh = 'soft'         #determines vicinity threshold type
soft_weight_thresh = 1e-3   #threshold for determining nonzero weights for SVDL

### --- Geometry/Problem Parameters --- ###
# Use-case Parameters - can ignore for now
phi_min = 0.            #min phi (x) val in GeV
phi_max = 3000.         #max phi (x) val in GeV
phi_bins = 300          #bins in phi (x)
omega_min = 0.          #min omega (y) val in MeV
omega_max = 2000./1000. #max omega (y) val in MeV
omega_bins = 200        #bins in omega (y)

# Gaussian Line Parameters
ngaus = 50              #num gaussians from which training data is created
ngausplot = 5           #num gaussians to plot/use in analysis
nsamp = 1000            #num samples in training gaussians
sigma_gaus = 0.0075      #std dev of training gaussians
val = 0.5               #specific value used for gaussian problems (circ/line)
xmin = 0.               #min xval of grid
xmax = 1.               #max xval of grid
xbins = 100             #bins in x
ymin = 0.               #min yval of grid
ymax = 1.               #max yval of grid
ybins = 100             #bins in y
xcov_change_linear_max_factor = 16.   #max change in growth in x
ycov_change_linear_max_factor = 16.   #max change in growth in y
#cov_change_skew_rate =             #radian angle rotation