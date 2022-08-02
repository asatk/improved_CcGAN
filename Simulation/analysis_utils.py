'''
Definitions of utility functions for analyzing CCGAN performance.

Author: Anthony Atkinson
'''

from models.CCGAN import generator
from matplotlib import pyplot as plt
import numpy as np
import torch

def sample_gen_for_label(gen: generator, n_samples: int, label: float, batch_size: int=500, n_dim: int=2) -> tuple[np.ndarray, np.ndarray]:
    '''
    label: normalized label in [0,1]
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim_gan = 2

    if batch_size>n_samples:
        batch_size = n_samples
    
    fake_samples = np.empty((0, n_dim), dtype=np.float)
    fake_labels = np.ones(n_samples) * label
    
    gen=gen.to(device)
    gen.eval()
    
    with torch.no_grad():
        sample_count = 0
        while sample_count < n_samples:
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)
            y = np.ones(batch_size) * label
            y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
            
            batch_fake_samples = gen(z, y)
            fake_samples = np.concatenate((fake_samples, batch_fake_samples.cpu().detach().numpy()))
            
            sample_count += batch_size

            if sample_count + batch_size > n_samples:
                batch_size = n_samples - sample_count

    return fake_samples, fake_labels

def plot_histogram(data: np.ndarray, filename: str, vmin: float=0.0, vmax: float=1.0, title: str="", x_axis_label: str="", y_axis_label: str="", **kwargs) -> None:
    '''
    data: bins x bins histogram
    filename: file system location where the histogram image is output
    vmin: minimum of histogram
    vmax: maximum of histogram
    x_axis_label: string label of x axis
    y_axis_label: string label of y axis
    kwargs: plotting kws

    Returns: None. Saves a figure.
    '''

    if kwargs['cmap'] is None:
        cmap = 'inferno'
    if kwargs['origin'] is None:
        origin = 'lower'
    if kwargs['facecolor'] is None:
        facecolor = 'black'

    plt.figure(facecolor=facecolor)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    im = plt.imshow(data.T, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax, extent=plot_lims_fn().flatten())
    plt.gca().use_sticky_edges = False
    plt.margins(0.05)
    # plt.gca().set_facecolor('black')    #for inferno
    plt.colorbar(im, shrink=0.8)
    plt.savefig(filename)

def make_histogram(samples: np.ndarray, bins: float) -> np.ndarray:
    '''
    samples: nsamp array x ndim
    
    Returns: bins x bins histogram
    '''
    h, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, range=plot_lims_fn())
    return h

def plot_scatter(samples_list: np.ndarray, color_list: list[str], label_list: list[str], filename: str, **kwargs):
    '''
    samples_list: num of different groups of samples x nsamp array x ndim - list of samples
    colors_list: num of different groups of samples - list of colors
    label_list: num of different groups of samples - list of labels
    filename: file system location where the histogram image is output
    kwargs: plotting kws
    
    Returns: bins x bins histogram
    '''

    for i in range(len(samples_list)):
        plt.scatter(samples_list[i][:, 0], samples_list[i][:, 1], c=color_list[i], edgecolor='none', alpha=1, s=25, label=label_list[i])
    
    if kwargs['title'] is not None:
        plt.title(kwargs['title'])

    plt.xlim(plot_lims_fn()[0])
    plt.ylim(plot_lims_fn()[1])
    plt.margins(0.05)
    plt.savefig(filename)
