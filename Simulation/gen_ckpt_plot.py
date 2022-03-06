
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

wd = os.getcwd()
save_npy_folder = wd + "/output/ckpt_numpy/"
save_jpg_folder = wd + "/output/ckpt_jpg/"
os.makedirs(save_jpg_folder,exist_ok=True)

#plot parameters - parameters defining how we sampled the generator from its ckpt
n_gaussians_plot = 12
n_samp_per_gaussian_plot = 100
N_iter = 6000

in_file_fake = save_npy_folder + "N{:04d}_S{:04d}_{:d}_FAKE.npy".format(n_gaussians_plot, n_samp_per_gaussian_plot, N_iter)
in_file_real = save_npy_folder + "N{:04d}_S{:04d}_{:d}_REAL.npy".format(n_gaussians_plot, n_samp_per_gaussian_plot, N_iter)
out_file_jpg = save_jpg_folder + "N{:04d}_S{:04d}_{:d}_PLOT.jpg".format(n_gaussians_plot, n_samp_per_gaussian_plot, N_iter)


fake_samples = np.load(in_file_fake)
real_samples_plot = np.load(in_file_real)


# Plot Settings
plot_in_train = True
fig_size=7
point_size = 25

plt.switch_backend('agg')
mpl.style.use('seaborn')
plt.figure(figsize=(fig_size, fig_size), facecolor='w')
plt.grid(b=True)
plt.scatter(real_samples_plot[:, 0], real_samples_plot[:, 1], c='blue', edgecolor='none', alpha=0.5, s=point_size, label="Real samples")
plt.scatter(fake_samples[:, 0], fake_samples[:, 1], c='green', edgecolor='none', alpha=1, s=point_size, label="Fake samples")
plt.legend(loc=1)
plt.savefig(out_file_jpg)