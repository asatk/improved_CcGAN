import numpy as np
from PIL import Image
import torch
from defs_sim import *

def sample_real_gaussian(n_samples, labels, gaus_points, cov_mtxs):
    '''

    n_samp_per_gaussian: number of samples drawn from each gaussian
    labels: list of labels for which samples are made
    gaus_points:    point where the i-th label is sampled from a circular gaussian
    cov_mtxs:       spread of the i-th gaussian in x and y (2dim)

    '''

    dim = cov_mtxs[0].shape[0]

    # n_samples samples from 'dim'-dimension gaussian
    samples = np.empty((0, dim), dtype=float)
    # a labels corresponding to each sample taken
    sampled_labels = np.empty((0,), dtype=float)

    for i in range(len(gaus_points)):
        point_i = gaus_points[i]
        label_i = labels[i]
        cov_mtx_i = cov_mtxs[i]

        # print(cov_mtx_i)
        # print(point_i)

        samples_i = np.random.multivariate_normal(point_i, cov_mtx_i, size=n_samples)
        samples = np.concatenate((samples, samples_i), axis=0)

        sampled_labels_i = np.ones(n_samples) * label_i
        sampled_labels = np.concatenate((sampled_labels, sampled_labels_i), axis=0)

    return samples, sampled_labels

def train_labels_circle(n_train):
    return np.linspace(0, 2*np.pi, n_train, endpoint=False)

def train_labels_line_1D(n_train):
    return np.linspace(xmin, xmax, n_train, endpoint=False)

def test_labels_circle(n_test):
    return np.linspace(0, 2*np.pi, n_test, endpoint=False)

def test_labels_line_1D(n_test):
    return np.linspace(xmin, xmax, n_test, endpoint=False)

def normalize_labels_circle(labels):
    return np.divide(labels, 2*np.pi)

def normalize_labels_line_1D(labels):
    return np.divide(np.subtract(labels, xmin), (xmax - xmin))

def gaus_point_circle(labels, radius):
    return np.multiply([np.sin(labels), np.cos(labels)], radius).T

def gaus_point_line_1D(labels, yval):
    return np.stack((labels, yval * np.ones(len(labels))), axis=1)

def plot_lims_circle(radius):
    return ((radius * -1.1, radius * 1.1), (radius * -1.1, radius * 1.1)) 

def plot_lims_line_1D():
    return ((xmin * 1.1, xmax * 1.1), (ymin * 1.1, ymax * 1.1))

def sample_gen_for_label(gen, n_samples, label, path=None, batch_size = 500, n_dim=2):
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

    if path is not None:
        raw_fake_samples = (fake_samples*0.5+0.5)*255.0
        raw_fake_samples = raw_fake_samples.astype(np.uint8)
        for i in range(n_samples):
            filename = path + '/' + str(i) + '.jpg'
            im = Image.fromarray(raw_fake_samples[i][0], mode='L')
            im = im.save(filename)

    return fake_samples, fake_labels