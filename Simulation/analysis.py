'''
Analysis script for CCGAN output on 2-D gaussians.

Author: Anthony Atkinson
'''

from json import load
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from os import listdir, makedirs
import re
import ROOT
from sys import argv
import torch

from models.CCGAN import generator
import train_utils

# Load run (hyper)parameters
if len(argv) != 3:
    print("Two arguments must be supplied: python analysis.py <params.json> <Sim #>")

filename_params_json = argv[1]
sim = argv[2]
params = load(open(filename_params_json, "r"))

normalize_fn = train_utils.normalize_labels_line_1d
plot_lims_fn = train_utils.plot_lims_line_1d

run_dir = params['run_dir']
nsim = int(params['nsim'])

# Plotting settings
mpl.style.use('./CCGAN-seaborn.mplstyle')
plt.switch_backend('agg')
#choose to look at one or the other
ROOT.gStyle.SetOptStat(0) #1110 if in use else 0
ROOT.gStyle.SetOptFit(1111)  #1111 if in use else 0
ROOT.gROOT.ForceStyle()
x_axis_label = 'x var'
y_axis_label = 'y var'
load_data_dir = run_dir + 'saved_data/'
load_gan_dir = run_dir + 'saved_models/'
save_data_dir = run_dir + 'analysis_%s/'%(sim)

# Files to load
filename_samples = load_data_dir + 'samples_train_%s.npy'%(sim)
filename_labels = load_data_dir + 'labels_train_%s.npy'%(sim)

p = re.compile(r"^CCGAN.+sim_(\d+).pth$")
for f in listdir(load_gan_dir):
    m = p.match(f)
    if m and m.group(1) == sim:
        filename_gan = load_gan_dir + f

# Files to save
makedirs(save_data_dir, exist_ok=True)
filename_real_jpg = save_data_dir + 'real.jpg'
filename_real_one_jpg = save_data_dir + 'real_%02i.jpg'
filename_xreal_one_jpg = save_data_dir + 'x_%02i_real.jpg'
filename_yreal_one_jpg = save_data_dir + 'y_%02i_real.jpg'
filename_fake_jpg = save_data_dir + 'fake.jpg'
filename_fake_one_jpg = save_data_dir + 'fake_%02i.jpg'
filename_xfake_one_jpg = save_data_dir + 'x_%02i_fake.jpg'
filename_yfake_one_jpg = save_data_dir + 'y_%02i_fake.jpg'
filename_net_jpg = save_data_dir + 'net.jpg'
filename_scatter_jpg = save_data_dir + 'scatter.jpg'

# Hyperparameters
n_samples_train = int(params['n_samples_train'])
n_gaussians = int(params['n_gaussians'])
n_gaussians_plot = int(params['n_gaussians_plot'])
fake_sample_scale = 1
n_samples_fake = fake_sample_scale * n_samples_train

# Histogram settings
xlo = params['xmin']
xhi = params['xmax']
xbins = int(params['xbins'])
ylo = params['ymin']
yhi = params['ymax']
ybins = int(params['ybins'])
xguess = 0.5

# Load training data and associated labels
samples_train = np.load(filename_samples)
labels = np.load(filename_labels)

dim = params['dim_gan']

# Load network at its most recent state
checkpoint = torch.load(filename_gan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = generator().to(device)
gen.load_state_dict(checkpoint['gen_state_dict'])

# Generate fake labels from network
labels_norm = normalize_fn(labels)
fake_samples = np.empty((0, dim), dtype=float)
for i in range(n_gaussians):
    
    label_i = labels_norm[i * n_samples_train]

    fake_samples_i, _ = train_utils.sample_gen_for_label(gen, n_samples_fake, label_i, batch_size=n_samples_fake)
    fake_samples = np.concatenate((fake_samples, fake_samples_i), axis=0)

# Select out certain gaussian and its samples for plotting purposes
plot_idxs = np.linspace(0, n_gaussians, n_gaussians_plot + 1, dtype=int, endpoint=False)[1:]
samples_train_plot = samples_train.reshape(n_gaussians, -1, 2)[plot_idxs]
samples_train_plot = samples_train_plot.reshape(n_gaussians_plot * n_samples_train, 2)
samples_train_plot_one = samples_train_plot[0:n_samples_train]

fake_samples_plot = fake_samples.reshape(n_gaussians, -1, 2)[plot_idxs]
fake_samples_plot = fake_samples_plot.reshape(n_gaussians_plot * n_samples_fake, 2)
fake_samples_plot_one = fake_samples_plot[0:n_samples_fake]

# Get real samples histogram
h_real, xedges, yedges = np.histogram2d(samples_train_plot[:, 0], samples_train_plot[:, 1], bins=xbins, range=plot_lims_fn())

# Get fake samples histogram
h_fake, _, _ = np.histogram2d(fake_samples_plot[:, 0], fake_samples_plot[:, 1], bins=xbins, range=plot_lims_fn())

# Get net histogram from real and fake
h_res = h_real - h_fake

# Max value across all distributions (for plotting and scaling)
vmin = 0.0
vmax = max(np.max(h_real), np.max(h_fake))

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

canv = ROOT.TCanvas("canv", "title", 800, 800)
canv.DrawCrosshair()

# Fit Function
# ref: https://root.cern.ch/doc/v608/group__PdfFunc.html#ga118e731634d25ce7716c0b279c5e6a16
# ref: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
xygaus_formula = "bigaus"
gaus2d = ROOT.TF2("gaus2d", xygaus_formula, xlo, xhi, ylo, yhi)
gaus2d.SetNpx(xbins)
gaus2d.SetNpy(ybins)
gaus2d.SetParameters(1., xguess, 0.075, 0.5, 0.075)
#only one gaus for both since same axes and statistics both ways - keep in mind
gaus1d = ROOT.TF1("gaus1d", "gaus", xlo, xhi)
gaus1d.SetNpx(xbins)
gaus1d.SetParameters(1., xguess, 0.075)
fit_options = "SMLRQ0"

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

hist_real_title_name = "Gaussian %i/%i real samples"
hist_fake_title_name = "Gaussian %i/%i fake samples"
hist = ROOT.TH2D("hist", "hist", xbins, xlo, xhi, ybins, ylo, yhi)
hist.SetMaximum(vmax)
histx = ROOT.TH1D("histx", "histx", xbins, xlo, xhi)
histx.SetLineWidth(2)
histy = ROOT.TH1D("histy", "histy", ybins, ylo, yhi)
histy.SetLineWidth(2)

# Plot Training Samples
for i in range(n_gaussians_plot):
    
    hist.SetTitle(hist_real_title_name%(i + 1, n_gaussians_plot))
    histx.SetTitle(hist_real_title_name%(i + 1, n_gaussians_plot) + " (X)")
    histy.SetTitle(hist_real_title_name%(i + 1, n_gaussians_plot) + " (Y)")

    samples_train_plot_one = samples_train_plot[i * n_samples_train: (i + 1) * n_samples_train]

    # Fill ROOT hist with every real sample
    for p in samples_train_plot_one:
        x = p[0]
        y = p[1]
        hist.Fill(x, y)

    # total profile and 2d gaus fit
    hist.Fit("gaus2d", fit_options)
    hist.Draw("COLZ")
    canv.Update()
    canv.SaveAs(filename_real_one_jpg%(i + 1))
    canv.Clear()

    # X projection and 1d gaus fit
    hist.ProjectionX("histx")
    histx.Fit("gaus1d", fit_options)
    histx.Draw("HIST")
    canv.Update()
    canv.SaveAs(filename_xreal_one_jpg%(i + 1))
    canv.Clear()
    gaus1d.SetParameters(1., xguess, 0.075)

    # Y projection and 1d gaus fit
    hist.ProjectionY("histy")
    histy.Fit("gaus1d", fit_options)
    histy.Draw("HIST")
    canv.Update()
    canv.SaveAs(filename_yreal_one_jpg%(i + 1))
    canv.Clear()

    # reset functions and histogram after every fit
    hist.Reset("ICES")
    gaus2d.SetParameters(1., xguess, 0.075, 0.5, 0.075)
    gaus1d.SetParameters(1., xguess, 0.075)

# Plot Fake Samples
for j in range(n_gaussians_plot):
    
    hist.SetTitle(hist_fake_title_name%(j + 1, n_gaussians_plot))
    histx.SetTitle(hist_fake_title_name%(j + 1, n_gaussians_plot) + " (X)")
    histy.SetTitle(hist_fake_title_name%(j + 1, n_gaussians_plot) + " (Y)")

    fake_samples_plot_one = fake_samples_plot[j * n_samples_fake: (j + 1) * n_samples_fake]
    
    # Fill ROOT hist with every fake sample
    for p in fake_samples_plot_one:
        x = p[0]
        y = p[1]
        hist.Fill(x, y)

    # total profile and 2d gaus fit
    hist.Fit("gaus2d", fit_options)
    hist.Draw("COLZ")
    canv.Update()
    canv.SaveAs(filename_fake_one_jpg%(j + 1))
    canv.Clear()

    # X projection and 1d gaus fit
    hist.ProjectionX("histx")
    histx.Fit("gaus1d", fit_options)
    histx.Draw("HIST")
    canv.Update()
    canv.SaveAs(filename_xfake_one_jpg%(j + 1))
    canv.Clear()
    gaus1d.SetParameters(1., xguess, 0.075)

    # Y projection and 1d gaus fit
    hist.ProjectionY("histy")
    histy.Fit("gaus1d", fit_options)
    histy.Draw("HIST")
    canv.Update()
    canv.SaveAs(filename_yfake_one_jpg%(j + 1))
    canv.Clear()

    # reset functions and histogram after every fit
    hist.Reset("ICES")
    gaus2d.SetParameters(1., xguess, 0.075, 0.5, 0.075)
    gaus1d.SetParameters(1., xguess, 0.075)