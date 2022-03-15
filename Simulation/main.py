'''

2D-Gaussian Simulation

'''

print("\n==================================================================================================")

import argparse
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import random
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import timeit

from opts import parse_opts
args = parse_opts()
wd = args.root_path
os.chdir(wd)

from models.cont_cond_GAN import cont_cond_discriminator
from models.cont_cond_GAN import cont_cond_generator
from Train_CcGAN import *
from train_utils import gaus_point_circle, normalize_labels_circle, sample_real_gaussian
from analysis_utils import l2_analysis, plot_analysis, two_was_analysis

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

#--------------------------------
# Extra Data Generation Settings
n_gaussians = args.n_gaussians
n_gaussians_eval = args.n_gaussians_eval
n_features = 2 # 2-D
radius = args.radius
# angles for training
angle_grid_train = np.linspace(0, 2*np.pi, n_gaussians, endpoint=False)
# angles for evaluation
unseen_angles_all = np.linspace(0, 2*np.pi, n_gaussians*100, endpoint=False)
unseen_angles_all = np.setdiff1d(unseen_angles_all, angle_grid_train)
angle_grid_eval = np.zeros(args.n_gaussians_eval)
for i in range(args.n_gaussians_eval):
    quantile_i = (i+1)/args.n_gaussians_eval
    angle_grid_eval[i] = np.quantile(unseen_angles_all, quantile_i, interpolation='nearest')
# angles for plotting
unseen_angles_all = np.linspace(0, 2*np.pi, n_gaussians*100, endpoint=False)
unseen_angles_all = np.setdiff1d(unseen_angles_all, angle_grid_train)
unseen_angle_grid_plot = np.zeros(args.n_gaussians_plot)
for i in range(args.n_gaussians_plot):
    quantile_i = (i+1)/args.n_gaussians_plot
    unseen_angle_grid_plot[i] = np.quantile(unseen_angles_all, quantile_i, interpolation='nearest')
# standard deviation of each Gaussian
sigma_gaussian = args.sigma_gaussian
### threshold to determine high quality samples
quality_threshold = sigma_gaussian*4 #good samples are within 5 standard deviation
print("Quality threshold is {}".format(quality_threshold))

#-------------------------------
# Plot Settings
plot_in_train = True
fig_size=7
point_size = 25

#-------------------------------
# output folders
save_models_folder = wd + '/output/saved_models/'
os.makedirs(save_models_folder,exist_ok=True)
save_images_folder = wd + '/output/saved_images/'
os.makedirs(save_images_folder,exist_ok=True)

#######################################################################################
'''                               Start Experiment                                 '''
#######################################################################################

def normalize_labels(labels):
    return normalize_labels_circle(labels)

def gaus_point(labels):
    return gaus_point_circle(labels, radius)

prop_recovered_modes = np.zeros(args.nsim) # num of recovered modes diveded by num of modes
prop_good_samples = np.zeros(args.nsim) # num of good fake samples diveded by num of all fake samples
avg_two_w_dist = np.zeros(args.nsim)

print("\n Begin The Experiment; Start Training {} >>>".format(args.GAN))
start = timeit.default_timer()
for nSim in range(args.nsim):
    print("Round %s" % (nSim))
    np.random.seed(nSim) #set seed for current simulation

    ###############################################################################
    # Data generation and dataloaders
    ###############################################################################
    n_samples_train = args.n_samp_per_gaussian_train
    labels_train = angle_grid_train
    gaus_points_train = gaus_point(angle_grid_train)
    gaus_points_train_plot = gaus_point(unseen_angle_grid_plot)
    
    #covariance matrix for each point sampled
    cov_mtxs_train = [sigma_gaussian**2 * np.eye(2)] * len(gaus_points_train)
    
    #this angles_train is not normalized; normalize if args.GAN is not cGAN.
    samples_train, sampled_labels_train = sample_real_gaussian(n_samples_train, labels_train, gaus_points_train, cov_mtxs_train) 
    samples_train_plot, _ = sample_real_gaussian(10, labels_train, gaus_points_train_plot, cov_mtxs_train)

    # plot training samples and their theoretical means
    filename_tmp = save_images_folder + 'samples_train_with_means_nSim_' + str(nSim) + '.jpg'
    if not os.path.isfile(filename_tmp):
        plt.switch_backend('agg')
        mpl.style.use('seaborn')
        plt.figure(figsize=(fig_size, fig_size), facecolor='w')
        plt.grid(b=True)
        plt.scatter(samples_train_plot[:, 0], samples_train_plot[:, 1], c='blue', edgecolor='none', alpha=0.5, s=point_size, label="Real samples")
        plt.scatter(gaus_points_train_plot[:, 0], gaus_points_train_plot[:, 1], c='red', edgecolor='none', alpha=1, s=point_size, label="Means")
        plt.legend(loc=1)
        plt.savefig(filename_tmp)

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
    print("{}/{}, {}, Sigma is {}, Kappa is {}".format(nSim+1, args.nsim, args.threshold_type, args.kernel_sigma, args.kappa))

    if args.GAN == 'CcGAN':
        save_GANimages_InTrain_folder = wd + '/output/saved_images/{}_{}_{}_{}_nSim_{}_InTrain'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa, nSim)
    os.makedirs(save_GANimages_InTrain_folder,exist_ok=True)

    #----------------------------------------------
    # Continuous cGAN
    Filename_GAN = save_models_folder + '/ckpt_CCGAN_niters_{}_seed_{}_{}_{}_{}_nSim_{}.pth'.format(args.niters_gan, args.seed, args.threshold_type, args.kernel_sigma, args.kappa, nSim)

    if not os.path.isfile(Filename_GAN):
        netG = cont_cond_generator(ngpu=NGPU, nz=args.dim_gan, out_dim=n_features, radius=radius)
        netD = cont_cond_discriminator(ngpu=NGPU, input_dim = n_features, radius=radius)

        # Start training
        netG, netD = train_CcGAN(args.kernel_sigma, args.kappa, samples_train, sampled_labels_train_norm, netG, netD, save_images_folder=save_GANimages_InTrain_folder, save_models_folder = save_models_folder, plot_in_train=plot_in_train, samples_tar_eval = samples_train_plot, angle_grid_eval = unseen_angle_grid_plot, fig_size=fig_size, point_size=point_size)

        # store model
        torch.save({
            'netG_state_dict': netG.state_dict(),
        }, Filename_GAN)
    else:
        print("Loading pre-trained generator >>>")
        checkpoint = torch.load(Filename_GAN)
        netG = cont_cond_generator(ngpu=NGPU, nz=args.dim_gan, out_dim=n_features, radius=radius).to(device)
        netG.load_state_dict(checkpoint['netG_state_dict'])

    ###############################################################################
    # Evaluation
    ###############################################################################
    if args.eval:
        print("\n Start evaluation >>>")

        # L2 Distance between real and fake samples
        n_samples_l2 = args.n_samp_per_gaussian_eval
        labels_l2_norm = normalize_labels(angle_grid_eval)
        gaus_points_l2 = gaus_point(angle_grid_eval)

        # percentage of high quality and recovered modes by taking l2 distance between real and fake samples
        prop_recovered_modes[nSim], prop_good_samples[nSim] = \
            l2_analysis(netG, n_samples_l2, labels_l2_norm, gaus_points_l2, quality_threshold)

        # 2-Wasserstein Distance
        n_samples_two_was = args.n_samp_per_gaussian_eval
        labels_two_was_norm = normalize_labels(angle_grid_eval)
        gaus_points_two_was = gaus_point(angle_grid_eval)
        cov_mtxs_two_was = [sigma_gaussian**2 * np.eye(2)] * len(labels_two_was_norm)

        avg_two_w_dist[nSim] = \
            two_was_analysis(netG, n_samples_two_was, labels_two_was_norm, gaus_points_two_was, cov_mtxs_two_was)

        # visualize fake samples
        filename_plot = save_images_folder + 'CCGAN_real_fake_samples_{}_sigma_{}_kappa_{}_nSim_{}.jpg'.format(args.threshold_type, args.kernel_sigma, args.kappa, nSim)
        n_gaussians_plot = args.n_gaussians_plot
        n_samples_plot = args.n_samp_per_gaussian_plot
        labels_plot = unseen_angle_grid_plot  #these dont matter
        gaus_points_plot = gaus_point(unseen_angle_grid_plot)
        cov_mtxs_plot = [sigma_gaussian**2 * np.eye(2)] * len(gaus_points_plot)

        plot_analysis(netG, n_samples_plot, n_gaussians_plot, labels_plot, gaus_points_plot, cov_mtxs_plot, normalize_labels, filename=filename_plot)
        
stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))
print("\n {}, Sigma is {}, Kappa is {}".format(args.threshold_type, args.kernel_sigma, args.kappa))
print("\n Prop. of good quality samples>>>\n")
print(prop_good_samples)
print("\n Prop. good samples over %d Sims: %.1f (%.1f)" % (args.nsim, np.mean(prop_good_samples), np.std(prop_good_samples)))
print("\n Prop. of recovered modes>>>\n")
print(prop_recovered_modes)
print("\n Prop. recovered modes over %d Sims: %.1f (%.1f)" % (args.nsim, np.mean(prop_recovered_modes), np.std(prop_recovered_modes)))
print("\r 2-Wasserstein Distance: %.2e (%.2e)"% (np.mean(avg_two_w_dist), np.std(avg_two_w_dist)))
print(avg_two_w_dist)
print("\n===================================================================================================")
