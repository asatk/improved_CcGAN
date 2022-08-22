"""
This module holds definitions of (hyper)parameters used to define the
geometry and data of the training run.
"""

from typing import Any

class Run():

    def __init__(self, params: dict[str, Any], name: str="", memo: str=""):

        self.name = name
        self.memo = memo

        ### --- Network/Run parameters --- ###
        self.root_path = params['root_path']        #absolute path to working directory containing project details
        self.seed = params['seed']                  #seed for RNG
        self.nsim = params['nsim']                  #num simulations in a run
        self.niter = params['niter']                #num iterations for which a simulation runs
        self.niter_resume = params['niter_resume']  #num iterations to resume training on a trained GAN
        self.niter_save = params['niter_save']      #num iterations to save GAN state
        self.nbatch_d = params['nbatch_d']          #num labels in dis batch
        self.nbatch_g = params['nbatch_g']          #num labels in gen batch
        self.sigma_kernel = params['sigma_kernel']  #defines width of kernel/noise for training
        self.kappa = params['kappa']                #vicinity parameter (for both hard and soft) - determines mk if negative
        self.dim_gan = params['dim_gan']            #dimensions of the GAN, i.e., 2 for 2-D problem - also referred to as latent dimension of GAN? understand what that means
        self.lr = params['lr']                      #learning rate: keep btwn [1e-3, 1e-6]
        self.geo = params['geo']                    #determines geometry/problem
        self.thresh = params['thresh']              #determines vicinity threshold type
        self.soft_weight_thresh = params['soft_weight_thresh']   #threshold for determining nonzero weights for SVDL

        ### --- Geometry/Problem Parameters --- ###
        # Use-case Parameters - can ignore for now
        self.phi_min = params['phi_min']            #min phi (x) val in GeV
        self.phi_max = params['phi_max']            #max phi (x) val in GeV
        self.phi_bins = params['phi_bins']          #bins in phi (x)
        self.omega_min = params['omega_min']        #min omega (y) val in MeV
        self.omega_max = params['omega_max']        #max omega (y) val in MeV
        self.omega_bins = params['omega_bins']      #bins in omega (y)

        # Gaussian Line Parameters
        self.ngaus = params['ngaus']                #num gaussians from which training data is created
        self.ngausplot = params['ngausplot']        #num gaussians to plot/use in analysis
        self.nsamp = params['nsamp']                #num samples in training gaussians
        self.sigma_gaus = params['sigma_gaus']      #std dev of training gaussians
        self.val = params['val']                    #specific value used for gaussian problems (circ/line)
        self.xmin = params['xmin']                  #min xval of grid
        self.xmax = params['xmax']                  #max xval of grid
        self.xbins = params['xbins']                #bins in x
        self.ymin = params['ymin']                  #min yval of grid
        self.ymax = params['ymax']                  #max yval of grid
        self.ybins = params['ybins']                #bins in y
        self.xcov_change_linear_max_factor = params['xcov_change_linear_max_factor']   #max change in growth in x
        self.ycov_change_linear_max_factor = params['ycov_change_linear_max_factor']   #max change in growth in y
        self.cov_change_skew_rate = params['cov_change_skew_rate']            #radian angle rotation

