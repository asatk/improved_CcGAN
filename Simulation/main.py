'''

2D-Gaussian Simulation

'''

import datetime
from json import dump
import numpy as np
import os
import random
import re
import torch
import torch.backends.cudnn as cudnn
import timeit

from opts import parse_opts
args = parse_opts()
wd = args.root_path
os.chdir(wd)

from models.cont_cond_GAN import cont_cond_discriminator
from models.cont_cond_GAN import cont_cond_generator
from Train_CcGAN import train_CCGAN
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
runs = np.empty((0,), dtype=int)
p = re.compile(r"run_(\d+)")
for d in os.listdir(wd + '/output/'):
    m = p.match(d)
    if m:
        runs = np.append(runs, int(m.group(1)))

if len(runs) == 0:
    current_run_dir = wd + '/output/run_0/'
else:
    all = np.linspace(0, np.max(runs) + 1, np.max(runs) + 2, dtype=int)
    diff = np.setdiff1d(all, runs)
    current_run_dir = wd + '/output/run_%i/'%(np.min(diff))

save_models_dir = current_run_dir + 'saved_models/'
os.makedirs(save_models_dir,exist_ok=True)
save_images_dir = current_run_dir + 'saved_images/'
os.makedirs(save_images_dir,exist_ok=True)
save_data_dir = current_run_dir + 'saved_data/'
os.makedirs(save_data_dir,exist_ok=True)

dict_params = {
    "date": str(datetime.datetime.now()),
    "seed": args.seed,
    "nsim": args.nsim,
    "niters": args.niters_gan,
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
    "xmin": defs.xmin,
    "xmax": defs.xmax,
    "xbins": defs.xbins,
    "ymin": defs.ymin,
    "ymax": defs.ymax,
    "ybins": defs.ybins,
    "xcov_change_linear_max_factor": defs.xcov_change_linear_max_factor,
    "ycov_change_linear_max_factor": defs.ycov_change_linear_max_factor,
    "run_dir": current_run_dir
}

with open(current_run_dir + "run_parameters.json", 'w+') as filename_params_json:
    dump(dict_params, filename_params_json, indent=4)

#######################################################################################
'''                               Start Experiment                                 '''
#######################################################################################

log = current_run_dir + "log.txt"
log_file = open(log, 'w+')

print("==================================================================================================")
print("\nBegin The Experiment; Start Training (geo: {})>>>".format(args.geo))
print("==================================================================================================", file=log_file)
print("\nBegin The Experiment; Start Training (geo: {})>>>".format(args.geo), file=log_file)
log_file.close()

start = timeit.default_timer()
for nSim in range(args.nsim):

    log_file = open(log, 'a+')

    print("\nSimulation %i" % (nSim))
    print("\nSimulation %i" % (nSim), file=log_file)

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

        print("Use rule-of-thumb formula to compute kernel_sigma >>>")
        print("Use rule-of-thumb formula to compute kernel_sigma >>>", file=log_file)

    if args.kappa < 0:
        kappa_base = np.abs(args.kappa)/args.n_gaussians

        if args.threshold_type=="hard":
            args.kappa = kappa_base
        else:
            args.kappa = 1/kappa_base**2

    ###############################################################################
    # Train a GAN model
    ###############################################################################

    print(("{}/{}, {}, Sigma is {:04f}, Kappa is {:04f}".format(nSim+1, args.nsim, args.threshold_type, args.kernel_sigma, args.kappa)))
    print(("{}/{}, {}, Sigma is {:04f}, Kappa is {:04f}".format(nSim+1, args.nsim, args.threshold_type, args.kernel_sigma, args.kappa)), file=log_file)
    log_file.close()

    #----------------------------------------------
    # Continuous cGAN
    Filename_GAN = save_models_dir + '/ckpt_CCGAN_niters_{}_seed_{}_{}_{:4f}_{:4f}_nSim_{}.pth'.format(args.niters_gan, args.seed, args.threshold_type, args.kernel_sigma, args.kappa, nSim)

    netG = cont_cond_generator(ngpu=NGPU, nz=args.dim_gan, out_dim=n_features, radius=radius)
    netD = cont_cond_discriminator(ngpu=NGPU, input_dim = n_features, radius=radius)

    # Start training
    netG, netD = train_CCGAN(args.kernel_sigma, args.kappa, samples_train, sampled_labels_train_norm, netG, netD, save_models_dir=save_models_dir, log=log)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
    }, Filename_GAN)
        
stop = timeit.default_timer()
log_file = open(log, 'a+')
print("GAN training finished; Time elapsed: {:04f}s".format(stop - start))
print("\n{}, Sigma is {:04f}, Kappa is {:04f}".format(args.threshold_type, args.kernel_sigma, args.kappa))
print("\n===================================================================================================")
print("GAN training finished; Time elapsed: {:04f}s".format(stop - start), file=log_file)
print("\n{}, Sigma is {:04f}, Kappa is {:04f}".format(args.threshold_type, args.kernel_sigma, args.kappa), file=log_file)
print("\n===================================================================================================", file=log_file)
log_file.close()