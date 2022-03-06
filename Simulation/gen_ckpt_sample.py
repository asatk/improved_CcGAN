"""
sample a generator in a saved state
"""

# import argparse
# import gc
# from main import Filename_GAN
# from Simulation.main import Filename_GAN
import numpy as np
import os
import sys
from tqdm import tqdm
import torch

# from opts import parse_opts
# args = parse_opts()

from utils import *
from models import *
from Train_CcGAN import *

wd = os.getcwd()
save_models_folder = wd + '/output/saved_models/'
save_npy_folder = wd + "/output/ckpt_numpy/"
os.makedirs(save_npy_folder,exist_ok=True)
os.chdir(wd)

# parameters used to define the checkpoint's state (the number and sampling parameters used to train)
if len(sys.argv) != 2:
    print("must (only) provide a GAN state ckpt file")
    exit

# Filename_GAN = sys.argv[1]
Filename_GAN = save_models_folder + "ckpt_CcGAN_niters_6000_seed_2020_hard_0.07410779864448126_0.016666666666666666_nSim_0.pth"
radius = 1.
sigma_gaussian = 0.02
dim_gan = 2
n_features = 2
n_gaussians = 120
# the parameters for gaussians that would be plotted but are instead saved
n_gaussians_plot = 12
n_samp_per_gaussian_plot = 100
N_iter = 6000

# angles for training
angle_grid_train = np.linspace(0, 2*np.pi, n_gaussians+1) # 12 clock is the start point; last angle is dropped to avoid overlapping.
angle_grid_train = angle_grid_train[0:n_gaussians]

# angles for plotting
unseen_angles_all = np.linspace(0, 2*np.pi, n_gaussians*100+1)
unseen_angles_all = np.setdiff1d(unseen_angles_all[0:n_gaussians*100], angle_grid_train)
unseen_angle_grid_plot = np.zeros(n_gaussians_plot)
for i in range(n_gaussians_plot):
    quantile_i = (i+1)/n_gaussians_plot
    unseen_angle_grid_plot[i] = np.quantile(unseen_angles_all, quantile_i, interpolation='nearest')

def generate_data(n_samp_per_gaussian, angle_grid):
    return sampler_CircleGaussian(n_samp_per_gaussian, angle_grid, radius = radius, sigma = sigma_gaussian, dim = n_features)

if os.path.isfile(Filename_GAN):
    print("Loading pre-trained generator >>>")
    checkpoint = torch.load(Filename_GAN)
    netG = cont_cond_generator(ngpu=NGPU, nz=dim_gan, out_dim=n_features, radius=radius).to(device)
    netG.load_state_dict(checkpoint['netG_state_dict'])

def fn_sampleGAN_given_label(nfake, label, batch_size):
    fake_samples, _ = SampCcGAN_given_label(netG, label, path=None, NFAKE = nfake, batch_size = batch_size)
    return fake_samples

fake_samples = np.zeros((n_gaussians_plot*n_samp_per_gaussian_plot, n_features))
for i_tmp in range(n_gaussians_plot):
    angle_curr = unseen_angle_grid_plot[i_tmp]
    fake_samples_curr = fn_sampleGAN_given_label(n_samp_per_gaussian_plot, angle_curr/(2*np.pi), batch_size=n_samp_per_gaussian_plot)
    if i_tmp == 0:
        fake_samples = fake_samples_curr
    else:
        fake_samples = np.concatenate((fake_samples, fake_samples_curr), axis=0)

real_samples_plot, _, _ = generate_data(n_samp_per_gaussian_plot, unseen_angle_grid_plot)

out_file_fake = save_npy_folder + "N{:04d}_S{:04d}_{:04d}_FAKE".format(n_gaussians_plot, n_samp_per_gaussian_plot, N_iter)
out_file_real = save_npy_folder + "N{:04d}_S{:04d}_{:04d}_REAL".format(n_gaussians_plot, n_samp_per_gaussian_plot, N_iter)

np.save(out_file_fake, fake_samples,allow_pickle=False)
np.save(out_file_real, real_samples_plot,allow_pickle=False)