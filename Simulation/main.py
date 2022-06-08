'''

2D-Gaussian Simulation

'''

print("\n==================================================================================================")

from json import dump
import numpy as np
import os
import random
import torch
import torch.backends.cudnn as cudnn
import timeit

from opts import parse_opts
args = parse_opts()
wd = args.root_path
os.chdir(wd)

from models.cont_cond_GAN import cont_cond_discriminator
from models.cont_cond_GAN import cont_cond_generator
from Train_CcGAN import *
from train_utils import *

#######################################################################################
'''                                   Settings                                      '''
#######################################################################################

#--------------------------------
# system
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCPU = 8

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#------------------------------------------------------------------------------
# Extensibility Hooks
# How to calculate labels and gaus peak points given the geometry of the
# problem (e.g. circle vs line)

def train_labels(n_train):
    if (args.geo == "circle"):
        return train_labels_circle(n_train)
    elif (args.geo == "line"):
        return train_labels_line_1d(n_train)

def test_labels(n_test):
    if (args.geo == "circle"):
        return test_labels_circle(n_test)
    elif (args.geo == "line"):
        return test_labels_line_1d(n_test)
    
def normalize_labels(labels):
    if (args.geo == "circle"):
        return normalize_labels_circle(labels)
    elif (args.geo == "line"):
        return normalize_labels_line_1d(labels)

def gaus_point(labels):
    if (args.geo == "circle"):
        return gaus_point_circle(labels, args.radius)
    elif (args.geo == "line"):
        return gaus_point_line_1d(labels, args.yval)

def plot_lims():
    if (args.geo == "circle"):
        return plot_lims_circle(radius=args.radius)
    elif (args.geo == "line"):
        return plot_lims_line_1d()

def cov_mtx(labels):
    return cov_change_const(labels, cov_xy(sigma_gaussian))

#--------------------------------
# Data Generation Settings
n_gaussians = args.n_gaussians
n_gaussians_eval = args.n_gaussians_eval
n_gaussians_plot = args.n_gaussians_plot

n_samples_train = args.n_samp_per_gaussian_train
n_samples_plot = args.n_samp_per_gaussian_plot

# standard deviation of each Gaussian
sigma_gaussian = args.sigma_gaussian
n_features = 2 # 2-D
radius = args.radius

#------------------------------------------------------------------------------
# Training and Testings Grids
test_label_grid_res = 100   # 100x more labels to test from than training data
# labels for training
labels_train = train_labels(n_gaussians)
# labels for evaluation
labels_test_all = test_labels(n_gaussians * test_label_grid_res)

labels_test_eval = np.empty((0,))
for i in range(n_gaussians_eval):
    quantile_i = (i+1)/n_gaussians_eval
    labels_test_eval = np.append(labels_test_eval, np.quantile(labels_test_all, quantile_i, interpolation='nearest'))
# labels for plotting
labels_test_plot = np.empty((0,))
for i in range(n_gaussians_plot):
    quantile_i = (i+1)/n_gaussians_plot
    labels_test_plot = np.append(labels_test_plot, np.quantile(labels_test_all, quantile_i, interpolation='nearest'))

### threshold to determine high quality samples
quality_threshold = sigma_gaussian*4 #good samples are within 4 standard deviation

#-------------------------------
# output folders
run_num = 0
for i, d in enumerate(os.listdir(wd + '/output/')):
    print(d[0:4])
    if d[0:4] == 'run_':
        print(i)
        run_num += 1

current_run_dir = wd + '/output/run_%i/'%(run_num)

save_models_dir = current_run_dir + 'saved_models/'
os.makedirs(save_models_dir,exist_ok=True)
save_images_dir = current_run_dir + 'saved_images/'
os.makedirs(save_images_dir,exist_ok=True)
save_data_dir = current_run_dir + 'saved_data/'
os.makedirs(save_data_dir,exist_ok=True)

dict_params = {
    "seed": args.seed,
    "niters": niters,
    "n_samples_train": n_samples_train,
    "n_gaussians": n_gaussians,
    "n_gaussians_plot": n_gaussians_plot,
    "sigma_gaussian": sigma_gaussian,
    "threshold_type": args.threshold_type,
    "geo": args.geo,
    "yval": args.yval,
    "radius": args.radius,
    "batch_size_disc": args.batch_size_disc,
    "batch_size_gene": args.batch_size_gene,
    "lr": args.lr_gan,
    "sigma": args.kernel_sigma,
    "kappa": args.kappa,
    "dim_gan": args.dim_gan,
    "xmin": xmin,
    "xmax": xmax,
    "xbins": xbins,
    "ymin": ymin,
    "ymax": ymax,
    "ybins": ybins,
    "xcov_change_linear_max_factor": xcov_change_linear_max_factor,
    "ycov_change_linear_max_factor": ycov_change_linear_max_factor,
    "run_dir": current_run_dir
}

with open(current_run_dir + "run_parameters.json", 'w+') as filename_params_json:
    dump(dict_params, filename_params_json, indent=4)

#######################################################################################
'''                               Start Experiment                                 '''
#######################################################################################

print("\n Begin The Experiment; Start Training (geo: {})>>>".format(args.geo))
start = timeit.default_timer()
for nSim in range(args.nsim):
    print("Simulation %i" % (nSim))
    np.random.seed(args.seed + nSim) #set seed for current simulation

    ###############################################################################
    # Data generation and dataloaders
    ###############################################################################
    
    # list of individual data points/labels being trained with
    gaus_points_train = gaus_point(labels_train)
    
    # covariance matrix for each point sampled
    cov_mtxs_train = cov_mtx(gaus_points_train)

    # samples from gaussian
    samples_train, sampled_labels_train = sample_real_gaussian(n_samples_train, labels_train, gaus_points_train, cov_mtxs_train)

    filename_samples_npy = save_data_dir + 'samples_train_' + str(nSim) + '.npy'
    filename_labels_npy = save_data_dir + 'labels_train_' + str(nSim) + '.npy'

    np.save(filename_samples_npy, samples_train, allow_pickle=False)
    np.save(filename_labels_npy, sampled_labels_train, allow_pickle=False)

    # preprocessing on labels
    sampled_labels_train_norm = normalize_labels(sampled_labels_train) #normalize to [0,1]

    # rule-of-thumb for the bandwidth selection
    if args.kernel_sigma<0:
        std_labels_train_norm = np.std(sampled_labels_train_norm)
        args.kernel_sigma = 1.06*std_labels_train_norm*(len(sampled_labels_train_norm))**(-1/5)
        print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")

    if args.kappa < 0:
        kappa_base = np.abs(args.kappa)/args.n_gaussians

        if args.threshold_type=="hard":
            args.kappa = kappa_base
        else:
            args.kappa = 1/kappa_base**2

    ###############################################################################
    # Train a GAN model
    ###############################################################################
    print("{}/{}, {}, Sigma is {:4f}, Kappa is {:4f}".format(nSim+1, args.nsim, args.threshold_type, args.kernel_sigma, args.kappa))

    #----------------------------------------------
    # Continuous cGAN
    Filename_GAN = save_models_dir + '/ckpt_CCGAN_niters_{}_seed_{}_{}_{:4f}_{:4f}_nSim_{}.pth'.format(args.niters_gan, args.seed, args.threshold_type, args.kernel_sigma, args.kappa, nSim)

    netG = cont_cond_generator(ngpu=NGPU, nz=args.dim_gan, out_dim=n_features, radius=radius)
    netD = cont_cond_discriminator(ngpu=NGPU, input_dim = n_features, radius=radius)

    # Start training
    netG, netD = train_CcGAN(args.kernel_sigma, args.kappa, samples_train, sampled_labels_train_norm, netG, netD, save_models_dir)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
    }, Filename_GAN)
        
stop = timeit.default_timer()
print("GAN training finished; Time elapsed: {}s".format(stop - start))
print("\n {}, Sigma is {}, Kappa is {}".format(args.threshold_type, args.kernel_sigma, args.kappa))
print("\n===================================================================================================")
