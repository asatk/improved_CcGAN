'''
Analysis script for CCGAN output on 2-D gaussians.

Author: Anthony Atkinson
'''

from json import load
import matplotlib as mpl
import matplotlib.pyplot as plt
from models.cont_cond_GAN import cont_cond_generator
import numpy as np
from os import listdir, makedirs
import re
import ROOT
from sys import argv
from train_utils import *
import torch

# Load run (hyper)parameters
if len(argv) != 3:
    print("Two arguments must be supplied: python analysis.py <params.json> <Sim #>")

filename_params_json = argv[1]
sim = argv[2]
params = load(open(filename_params_json, "r"))

normalize_fn = normalize_labels_line_1d
plot_lims_fn = plot_lims_line_1d

run_dir = params['run_dir']

nsim = int(params['nsim'])

# Plotting settings
mpl.style.use('./CCGAN-seaborn.mplstyle')
plt.switch_backend('agg')
ROOT.gStyle.SetOptFit(1)
x_axis_label = 'x var'
y_axis_label = 'y var'
load_data_dir = run_dir + 'saved_data/'
load_gan_dir = run_dir + 'saved_models/'
save_data_dir = run_dir + 'analysis/'

# Files to load
filename_samples = load_data_dir + 'samples_train_0.npy'
filename_labels = load_data_dir + 'labels_train_0.npy'

p = re.compile(r"^ckpt_CCGAN.+nSim_(\d+).pth$")
for f in listdir(load_gan_dir):
    m = p.match(f)
    if m and m.group(1) == sim:
        filename_GAN = load_gan_dir + f

# Files to save
makedirs(save_data_dir, exist_ok=True)
filename_real_jpg = save_data_dir + 'real.jpg'
filename_real_one_jpg = save_data_dir + 'real_%02i.jpg'
filename_fake_jpg = save_data_dir + 'fake.jpg'
filename_fake_one_jpg = save_data_dir + 'fake_%02i.jpg'
filename_net_jpg = save_data_dir + 'net.jpg'
filename_scatter_jpg = save_data_dir + 'scatter.jpg'

# Hyperparameters
n_samples_train = int(params['n_samples_train'])
n_gaussians = int(params['n_gaussians'])
n_gaussians_plot = int(params['n_gaussians_plot'])
fake_sample_scale = 1
n_samples_fake = fake_sample_scale * n_samples_train

# Load training data and associated labels
samples_train = np.load(filename_samples)
labels = np.load(filename_labels)

dim = params['dim_gan']

# Load network at its most recent state
checkpoint = torch.load(filename_GAN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = cont_cond_generator().to(device)
netG.load_state_dict(checkpoint['netG_state_dict'])

# Generate fake labels from network
labels_norm = normalize_fn(labels)
fake_samples = np.empty((0, dim), dtype=float)
for i in range(n_gaussians):
    
    label_i = labels_norm[i * n_samples_train]

    fake_samples_i, _ = sample_gen_for_label(netG, n_samples_fake, label_i, batch_size=n_samples_fake)
    fake_samples = np.concatenate((fake_samples, fake_samples_i), axis=0)

# Select out certain gaussian and its samples for plotting purposes
plot_idxs = np.linspace(0, n_gaussians, n_gaussians_plot + 1, dtype=int, endpoint=False)[1:]
samples_train_plot = samples_train.reshape(n_gaussians, -1, 2)[plot_idxs]
samples_train_plot = samples_train_plot.reshape(n_gaussians_plot * n_samples_train, 2)
samples_train_plot_one = samples_train_plot[0:n_samples_train]

fake_samples_plot = fake_samples.reshape(n_gaussians, -1, 2)[plot_idxs]
fake_samples_plot = fake_samples_plot.reshape(n_gaussians_plot * n_samples_fake, 2)
fake_samples_plot_one = fake_samples_plot[0:n_samples_fake]

# norm_factor = np.sum(np.histogram2d(samples_train_plot_one[:, 0], samples_train_plot_one[:, 1], bins=100, range=plot_lims_fn())[0])

# Get real samples histogram
h_real, xedges, yedges = np.histogram2d(samples_train_plot[:, 0], samples_train_plot[:, 1], bins=100, range=plot_lims_fn())
# h_real = np.divide(h_real, np.sum(h_real))
# h_real = np.divide(h_real, norm_factor)
# h_real_one, _, _ = np.histogram2d(samples_train_plot_one[:, 0], samples_train_plot_one[:, 1], bins=100, range=plot_lims_fn())

# Get fake samples histogram
h_fake, _, _ = np.histogram2d(fake_samples_plot[:, 0], fake_samples_plot[:, 1], bins=100, range=plot_lims_fn())
# h_fake = np.divide(h_fake, np.sum(h_fake))
# h_fake = np.divide(h_fake, norm_factor)
# h_fake_one, _, _ = np.histogram2d(fake_samples_plot_one[:, 0], fake_samples_plot_one[:, 1], bins=100, range=plot_lims_fn())

# Get net histogram from real and fake
h_res = h_real - h_fake

vmin = 0.0
vmax = max(np.max(h_real), np.max(h_fake))
print(vmax)

#Scatter of fake samples on real samples
plt.scatter(samples_train_plot[:, 0], samples_train_plot[:, 1], c='blue', edgecolor='none', alpha=0.5, s=25, label="Real samples")
plt.scatter(fake_samples_plot[:, 0], fake_samples_plot[:, 1], c='green', edgecolor='none', alpha=1, s=25, label="Fake samples")
plt.xlim(plot_lims_fn()[0])
plt.ylim(plot_lims_fn()[1])
plt.margins(0.05)
plt.savefig(filename_scatter_jpg)

plt.clf()

# Plot Real samples
plt.figure(facecolor='w')
plt.title("%i Training Samples"%(n_gaussians_plot))
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
im = plt.imshow(h_real.T, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax, extent=plot_lims_fn().flatten())
plt.gca().use_sticky_edges = False
plt.margins(0.05)
plt.gca().set_facecolor('black')
plt.colorbar(im, shrink=0.8)
plt.savefig(filename_real_jpg)

plt.clf()

# Plot Fake samples
plt.figure(facecolor='w')
plt.title("%i Generated Samples"%(n_gaussians_plot))
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
im = plt.imshow(h_fake.T, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax, extent=plot_lims_fn().flatten())
plt.gca().use_sticky_edges = False
plt.margins(0.05)
plt.gca().set_facecolor('black')
plt.colorbar(im, shrink=0.8)
plt.savefig(filename_fake_jpg)

plt.clf()

resmaxval = max(abs(h_res.min()), abs(h_res.max()))

# Plot net histogram
plt.figure(facecolor='w')
plt.title("Residual of %i Gaussians (Real - Fake)"%(n_gaussians_plot))
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
im = plt.imshow(h_res.T, cmap="RdBu_r", vmin=-1*resmaxval, vmax=resmaxval, origin='lower', extent=plot_lims_fn().flatten())
plt.gca().use_sticky_edges = False
plt.margins(0.05)
plt.gca().set_facecolor('w')
plt.colorbar(im, shrink=0.8)
plt.savefig(filename_net_jpg)

canv = ROOT.TCanvas("canv", "title", 1000, 1000)
canv.DrawCrosshair()

xlo = params['xmin']
xhi = params['xmax']
xbins = int(params['xbins'])
ylo = params['ymin']
yhi = params['ymax']
ybins = int(params['ybins'])
xguess = 0.5
# hist = ROOT.TH2D("hist", "hist title", xbins, xlo, xhi, ybins, ylo, yhi)
# histAll = ROOT.TH2D("histall", "all gauss real samples", xbins, xlo, xhi, ybins, ylo, yhi)

func = ROOT.TF2("func", "xygaus", xlo, xhi, ylo, yhi)
func.SetNpx(xbins)
func.SetNpy(ybins)
func.SetParameters(1., xguess, 0.075, 0.5, 0.075)

'''
S for Save function in histogram fit
M for search for more minima after initial fit
0 for don't plot
L for NLL minimization method instead of X^2
Q for Quiet
V for Verbose
R for Range defined in TF1 def
B for fixing parameters to those defined in fn pre-fit
'''
for i in range(n_gaussians_plot):
    hist = ROOT.TH2D("hist", "Gaussian %i/%i real samples"%(i + 1, n_gaussians_plot), xbins, xlo, xhi, ybins, ylo, yhi)
    samples_train_plot_one = samples_train_plot[i * n_samples_train: (i + 1) * n_samples_train]
    h_real_one, _, _ = np.histogram2d(samples_train_plot_one[:, 0], samples_train_plot_one[:, 1], bins=100, range=plot_lims_fn())
    # h_real_one = np.divide(h_real_one, np.max(h_real_one))
    for bin_y in range(ybins):  
        y = yedges[bin_y]
        for bin_x in range(xbins):
            x = xedges[bin_x]
            z = h_real_one[bin_x,bin_y]
            # don't add to array if no data
            if z != 0:
                hist.Fill(x, y, z)
                # histAll.Fill(x, y, z)
    hist.SetMaximum(vmax)
    hist.Fit("func", "SM0RQ")
    hist.Draw("COLZ")
    canv.Update()
    canv.SaveAs(filename_real_one_jpg%(i + 1))
    # input()
    hist.Delete()

# histAll.Draw("COLZ")
# canv.Update()
# canv.SaveAs()
# input()
# histAll.Delete()

# histAll = ROOT.TH2D("histall", "all gauss fake samples", xbins, xlo, xhi, ybins, ylo, yhi)

for j in range(n_gaussians_plot):
    hist = ROOT.TH2D("hist", "Gaussian %i/%i fake samples"%(j + 1, n_gaussians_plot), xbins, xlo, xhi, ybins, ylo, yhi)
    fake_samples_plot_one = fake_samples_plot[j * n_samples_fake: (j + 1) * n_samples_fake]
    h_fake_one, _, _ = np.histogram2d(fake_samples_plot_one[:, 0], fake_samples_plot_one[:, 1], bins=100, range=plot_lims_fn())
    # h_fake_one = np.divide(h_fake_one, np.max(h_fake_one))
    for bin_y in range(ybins):  
        y = yedges[bin_y]
        for bin_x in range(xbins):
            x = xedges[bin_x]
            z = h_fake_one[bin_x,bin_y]
            # don't add to array if no data
            if z != 0:
                hist.Fill(x, y, z)
                # histAll.Fill(x, y, z)
    hist.SetMaximum(vmax)
    hist.Fit("func", "SM0R")
    hist.Draw("COLZ")
    canv.Update()
    canv.SaveAs(filename_fake_one_jpg%(j + 1))
    # input()
    hist.Delete()

# histAll.Draw("COLZ")
# canv.Update()
# canv.SaveAs()
# # input()
# histAll.Delete()
