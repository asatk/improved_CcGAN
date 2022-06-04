import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
from train_utils import sample_gen_for_label
from train_utils import sample_real_gaussian
from utils import two_wasserstein

# def gaus_analysis():

#     def func(x, ):


#     curve_fit(f=func, xdata=, ydata=)

def l2_analysis(netG, n_samples, labels_norm, gaus_points, quality_threshold):
    
    prop_recovered_modes = 0
    l2_dis_fake_samples = np.empty((0,), dtype=float)
    dim = gaus_points.shape[1]

    # percentage of high quality and recovered modes
    for i in range(len(labels_norm)):
        label_norm_i = labels_norm[i]
        gaus_point_i = np.repeat(gaus_points[i].reshape(1, dim), n_samples, axis=0)
        
        fake_samples_i, _ = sample_gen_for_label(netG, n_samples, label_norm_i, batch_size=n_samples)

        #l2 distance between a fake sample and its mean
        l2_dis_fake_samples_i = np.sqrt(np.sum((fake_samples_i-gaus_point_i)**2, axis=1))
        l2_dis_fake_samples = np.concatenate((l2_dis_fake_samples, l2_dis_fake_samples_i))

        # whether this mode is recovered?
        if sum(l2_dis_fake_samples_i <= quality_threshold)>0:
            prop_recovered_modes += 1

    prop_recovered_modes = (prop_recovered_modes/len(labels_norm))*100
    #proportion of good fake samples
    prop_good_samples = sum(l2_dis_fake_samples<=quality_threshold)/len(l2_dis_fake_samples)*100 

    return prop_recovered_modes, prop_good_samples

def plot_analysis(netG, n_samples, n_gaussians, samples_real, labels, normalize_fn, plot_lims_fn, filename=None, fig_size=10, point_size=25):
    
    if filename == None:
        filename = "./plot_analysis.jpg"

    dim = samples_real.shape[-1]

    labels_norm = normalize_fn(labels)
    fake_samples = np.empty((0, dim), dtype=float)
    for i in range(n_gaussians):
        
        label_i = labels_norm[i]

        fake_samples_i, _ = sample_gen_for_label(netG, n_samples, label_i, batch_size=n_samples)
        fake_samples = np.concatenate((fake_samples, fake_samples_i), axis=0)
    
    # n_samples_plot = n_samples

    # plot_idxs = np.linspace(0, n_gaussians, n_gaussians_plot + 1, dtype=int, endpoint=False)[1:]
    
    # samples_real_plot, _ = sample_real_gaussian(n_samples_plot, labels_norm, gaus_points, cov_mtxs)

    mpl.style.use('./CCGAN-seaborn.mplstyle')
    plt.switch_backend('agg')
    plt.figure(figsize=(fig_size, fig_size), facecolor='w')
    plt.grid(b=False)
    plt.title("Generated Samples", fontsize=25)
    plt.xlim(plot_lims_fn()[0])
    plt.ylim(plot_lims_fn()[1])
    plt.use_sticky_edges = False
    # plt.margins(0.2, tight=False)
    plt.gca().set_xmargin(0.2)
    print(plt.xlim())
    print(plt.ylim())
    plt.xlabel('x var', fontsize=15)
    plt.ylabel('y var', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.scatter(samples_real_plot[:, 0], samples_real_plot[:, 1], c='blue', edgecolor='none', alpha=0.5, s=point_size, label="Real samples")
    # plt.scatter(fake_samples[:, 0], fake_samples[:, 1], c='green', edgecolor='none', alpha=1, s=point_size, label="Fake samples")
    h_fake, _, _ = np.histogram2d(fake_samples[:, 0], fake_samples[:, 1], bins=100, range=plot_lims_fn())
    im = plt.imshow(h_fake, cmap='inferno', origin='lower', extent=plot_lims_fn().flatten())
    plt.colorbar(im, shrink=0.8)
    # plt.legend(loc=1)
    plt.savefig(filename)
    
    h_real, _, _ = np.histogram2d(samples_real[:, 0], samples_real[:, 1], bins=100, range=plot_lims_fn())

    h_res = h_real - h_fake

    plt.clf()

    maxval = max(abs(h_res.min()), abs(h_res.max()))
    # minmaxval = min(abs(h_res.min()), abs(h_res.max()))

    ax = plt.gca()
    plt.grid(b=False)
    plt.title("Residual Counts (Real - Fake)")
    plt.xlim(plot_lims_fn()[0])
    plt.ylim(plot_lims_fn()[1])
    im = ax.imshow(h_res.T, cmap="coolwarm", vmin=-1*maxval, vmax=maxval, origin='lower', extent=plot_lims_fn().flatten())
    plt.colorbar(im, shrink=0.8)

    # plt.legend(loc=1)
    plt.savefig(filename[:-4] + "_RES" + ".jpg")

def two_was_analysis(netG, n_samples, labels_norm, gaus_points, cov_mtxs):
    two_w_dist_all = np.empty((0,1), dtype=float)
    for i in tqdm(range(len(labels_norm))):
        label_norm_i = labels_norm[i]
        gaus_point_i = gaus_points[i]
        cov_mtx_i = cov_mtxs[i]
        
        # sample from trained GAN
        fake_samples_i, _ = sample_gen_for_label(netG, n_samples, label_norm_i, batch_size=n_samples)

        # the sample mean and sample cov of fake samples with current label
        fake_point_mean_i = np.mean(fake_samples_i, axis = 0)
        fake_point_cov_i = np.cov(fake_samples_i.transpose())

        # 2-W distance for current label
        two_w_dist_i = two_wasserstein(gaus_point_i, fake_point_mean_i, cov_mtx_i, fake_point_cov_i, eps=1e-20)
        two_w_dist_all = np.append(two_w_dist_all, two_w_dist_i)
    
    #average over all evaluation angles
    avg_two_w_dist = sum(two_w_dist_all)/len(two_w_dist_all)
    return avg_two_w_dist
