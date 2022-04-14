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

from models.cont_cond_GAN import *

from train_utils import *
from defs_sim import *

geo = "line"
wd = os.getcwd()
save_models_folder = wd + '/output/saved_models_{}/'.format(geo)
save_npy_folder = wd + "/output/ckpt_numpy/"
os.makedirs(save_npy_folder,exist_ok=True)
os.chdir(wd)

# parameters used to define the checkpoint's state (the number and sampling parameters used to train)
if len(sys.argv) != 2:
    print("must (only) provide a GAN state ckpt file")
    exit

# Filename_GAN = sys.argv[1]
Filename_GAN = save_models_folder + "ckpt_CCGAN_niters_6000_seed_2020_hard_0.074108_0.016667_nSim_0.pth"
radius = 1.
yval = 0.5
sigma_gaussian = 0.0075
dim_gan = 2
n_features = 2
n_gaussians = 12
label_idxs = [3, 4, 5]
# the parameters for gaussians that would be plotted but are instead saved
# n_gaussians_plot = 12
n_samp_per_gaussian = 100
N_iter = 6000
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCPU = 8

out_file_fake = save_npy_folder + "{}_N{:04d}_S{:04d}_{:04d}_FAKE".format(geo, n_gaussians, n_samp_per_gaussian, N_iter)
out_file_real = save_npy_folder + "{}_N{:04d}_S{:04d}_{:04d}_REAL".format(geo, n_gaussians, n_samp_per_gaussian, N_iter)

# angles for training
def train_labels(n_train):
    if (geo == "circle"):
        return train_labels_circle(n_train)
    elif (geo == "line"):
        return train_labels_line_1D(n_train)

def test_labels(n_test):
    if (geo == "circle"):
        return test_labels_circle(n_test)
    elif (geo == "line"):
        return test_labels_line_1D(n_test)
    
def normalize_labels(labels):
    if (geo == "circle"):
        return normalize_labels_circle(labels)
    elif (geo == "line"):
        return normalize_labels_line_1D(labels)

def gaus_point(labels):
    if (geo == "circle"):
        return gaus_point_circle(labels, radius)
    elif (geo == "line"):
        return gaus_point_line_1D(labels, yval)

def plot_lims():
    if (geo == "circle"):
        return plot_lims_circle(radius=radius)
    elif (geo == "line"):
        return plot_lims_line_1D()

labels = train_labels(n_gaussians)
labels_selected = labels[label_idxs]
labels_norm = normalize_labels(labels_selected)
gaus_points = gaus_point(labels_selected)
cov_mtxs = [sigma_gaussian**2 * np.eye(2)] * len(gaus_points)


# gaus_points_train_plot = gaus_point(labels_test)

# angles for plotting

if os.path.isfile(Filename_GAN):
    print("Loading pre-trained generator >>>")
    checkpoint = torch.load(Filename_GAN)
    netG = cont_cond_generator(ngpu=NGPU, nz=dim_gan, out_dim=n_features, radius=radius).to(device)
    netG.load_state_dict(checkpoint['netG_state_dict'])

fake_samples = np.empty((0,2))

for i in range(len(gaus_points)):
    label = labels_norm[i]
    fake_samples_i, _ = sample_gen_for_label(netG, n_samp_per_gaussian, label, path=None)
    print(fake_samples_i)
    fake_samples = np.concatenate((fake_samples, fake_samples_i), axis=0)

real_samples, _ = sample_real_gaussian(n_samp_per_gaussian, labels_selected, gaus_points, cov_mtxs)

# print(fake_samples)
# print(real_samples)

# if (label_idx < 0):
#     fake_samples = np.zeros((n_gaussians*n_samp_per_gaussian, n_features))
#     for i_tmp in range(n_gaussians):
#         label = labels_train[i_tmp]
#         fake_samples_curr = sample_gen_for_label(netG, n_samp_per_gaussian, label, path=out_file_fake)
#         if i_tmp == 0:
#             fake_samples = fake_samples_curr
#         else:
#             fake_samples = np.concatenate((fake_samples, fake_samples_curr), axis=0)

#     real_samples_plot, _, _ = sample_real_gaussian(n_samp_per_gaussian, labels_train, gaus_points_train, cov_mtxs_train)
# else:
#     fake_samples = sample_gen_for_label(netG, n_samp_per_gaussian, labels_train[label_idx], path=out_file_fake) 
#     real_samples_plot, _, _ = sample_real_gaussian(n_samp_per_gaussian, np.array([labels_train[label_idx]]), np.array([gaus_points_train[label_idx]]), np.array([cov_mtxs_train[label_idx]]))


np.save(out_file_fake, fake_samples,allow_pickle=False)
np.save(out_file_real, real_samples,allow_pickle=False)