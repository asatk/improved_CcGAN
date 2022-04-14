
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

from defs_sim import *

wd = os.getcwd()
save_npy_folder = wd + "/output/ckpt_numpy/"
save_jpg_folder = wd + "/output/ckpt_jpg/"
os.makedirs(save_jpg_folder,exist_ok=True)

#plot parameters - parameters defining how we sampled the generator from its ckpt
geo = "line"
n_gaussians_plot = 12
n_samp_per_gaussian_plot = 100
N_iter = 6000

in_file_fake = save_npy_folder + "{}_N{:04d}_S{:04d}_{:d}_FAKE.npy".format(geo, n_gaussians_plot, n_samp_per_gaussian_plot, N_iter)
in_file_real = save_npy_folder + "{}_N{:04d}_S{:04d}_{:d}_REAL.npy".format(geo, n_gaussians_plot, n_samp_per_gaussian_plot, N_iter)
out_file_jpg = save_jpg_folder + "{}_N{:04d}_S{:04d}_{:d}_PLOT".format(geo, n_gaussians_plot, n_samp_per_gaussian_plot, N_iter)
out_file_fake = save_jpg_folder + "{}_N{:04d}_S{:04d}_{:d}_PLOT_F".format(geo, n_gaussians_plot, n_samp_per_gaussian_plot, N_iter)
out_file_real = save_jpg_folder + "{}_N{:04d}_S{:04d}_{:d}_PLOT_R".format(geo, n_gaussians_plot, n_samp_per_gaussian_plot, N_iter)

fake_samples = np.load(in_file_fake)
real_samples_plot = np.load(in_file_real)


# Plot Settings
plot_types = ["2dscatter", "3dhist", "2dcont"]
plot_type = plot_types[0]

plot_in_train = True
fig_size=7
point_size = 25
bins = (100,100)

def make_2d_scatter(data: np.ndarray, save_file: str, stack: bool = False, data2: np.ndarray = None):
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    plt.figure(figsize=(fig_size, fig_size), facecolor='w')
    plt.grid(b=True)
    plt.scatter(data[:, 0], data[:, 1], c='green', edgecolor='none', alpha=1, s=point_size, label="samples")
    if stack:
        plt.scatter(data2[:, 0], data2[:, 1], c='blue', edgecolor='none', alpha=1, s=point_size, label="real samples")
    # else:
        

    
    plt.legend(loc=1)

    if save_file is not None:
        plt.savefig(save_file+"_%s.jpg"%(plot_type + ("_STACK" if stack else "")))

def make_3d_histogram(data: np.ndarray, save_file: str, stack: bool = False):
    if not stack:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")

    x_pt = data[:,0]
    y_pt = data[:,1]

    hist, xedges, yedges = np.histogram2d(x_pt, y_pt, bins=bins)
    # hist = hist.T
    x, y = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

    x = x.flatten()/2.
    y = y.flatten()/2.
    z = np.zeros_like(x)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = hist.flatten()

    ax.bar3d(x,y,z,dx,dy,dz)
    plt.savefig(save_file+"_%s.jpg"%(plot_type + ("_STACK" if stack else "")))

def make_2d_contour(data: np.ndarray, save_file: str, stack: bool = False):
    if not stack:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    x_pt = data[:,0]
    y_pt = data[:,1]

    hist, xedges, yedges = np.histogram2d(x_pt, y_pt, bins=bins)
    hist = hist.T
    x, y = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

    x = x.flatten()/2.
    y = y.flatten()/2.
    
    ax.contour(xedges[1:], yedges[1:], hist, 50)

    plt.savefig(save_file+"_%s.jpg"%(plot_type + ("_STACK" if stack else "")))
    plt.imsave(save_file+"_%s.png"%(plot_type + ("_STACK" if stack else "")),
            hist, cmap="gray", vmin=0., vmax=np.max(hist), format="png", origin="lower")


if plot_type == "2dscatter":
    make_2d_scatter(fake_samples, out_file_fake, stack=True, data2=real_samples_plot)
    make_2d_scatter(fake_samples, out_file_real, False)
    make_2d_scatter(real_samples_plot, out_file_real, False)

elif plot_type == "3dhist":
    make_3d_histogram(fake_samples, out_file_fake, False)
    make_3d_histogram(real_samples_plot, out_file_real, False)

elif plot_type == "2dcont":
    make_2d_contour(fake_samples, out_file_fake, False)
    make_2d_contour(real_samples_plot, out_file_real, False)


    