'''
Definitions of utility functions for training CCGAN.

Author: Anthony Atkinson
'''

import numpy as np
import defs

def sample_real_gaussian(n_samples: int, labels: np.ndarray, gaus_points: np.ndarray, cov_mtxs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
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

        samples_i = np.random.multivariate_normal(point_i, cov_mtx_i, size=n_samples)
        samples = np.concatenate((samples, samples_i), axis=0)

        sampled_labels_i = np.ones(n_samples) * label_i
        sampled_labels = np.concatenate((sampled_labels, sampled_labels_i), axis=0)

    return samples, sampled_labels

def train_labels_circle(n_train: int) -> np.ndarray:
    return np.linspace(0, 2*np.pi, n_train, endpoint=False)

def train_labels_line_1d(n_train: int) -> np.ndarray:
    return np.linspace(defs.xmin, defs.xmax, n_train, endpoint=False)

def test_labels_circle(n_test: int) -> np.ndarray:
    return np.linspace(0, 2*np.pi, n_test, endpoint=False)

def test_labels_line_1d(n_test: int) -> np.ndarray:
    return np.linspace(defs.xmin, defs.xmax, n_test, endpoint=False)

def normalize_labels_circle(labels: np.ndarray) -> np.ndarray:
    return np.divide(labels, 2*np.pi)

def recover_labels_circle(labels: np.ndarray) -> np.ndarray:
    return np.multiply(labels, 2*np.pi)

def normalize_labels_line_1d(labels: np.ndarray) -> np.ndarray:
    return np.divide(np.subtract(labels, defs.xmin), (defs.xmax - defs.xmin))

def recover_labels_line_1d(labels: np.ndarray) -> np.ndarray:
    return np.add(np.multiply(labels, (defs.xmax - defs.xmin)), defs.xmax)

def gaus_point_circle(labels: np.ndarray, radius: float) -> np.ndarray:
    return np.multiply([np.sin(labels), np.cos(labels)], radius).T

def gaus_point_line_1d(labels: np.ndarray, yval: float) -> np.ndarray:
    return np.stack((labels, yval * np.ones(len(labels))), axis=1)

def plot_lims_circle(radius: float) -> np.ndarray:
    return np.multiply(np.ones((2,2)), radius)

def plot_lims_line_1d() -> np.ndarray:
    return np.array([[defs.xmin, defs.xmax], [defs.ymin, defs.ymax]])

def cov_xy(sigma1: float, sigma2: float=None) -> np.ndarray:
    if sigma2 is None:
        sigma2 = sigma1
    return np.array([[sigma1**2, 0.],[0., sigma2**2]])

def cov_skew(cov11: float, cov12: float, cov21: float=None, cov22: float=None) -> np.ndarray:
    if cov21 is None:
        cov21 = cov12
    if cov22 is None:
        cov22 = cov11
    return np.array([[cov11**2, cov12**2],[cov21**2, cov22**2]])
    
def cov_change_const(labels: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return np.repeat([cov], len(labels), axis=0)

def cov_change_linear(labels: np.ndarray, cov: np.ndarray) -> list[np.ndarray]:
    return [cov * 
        (1 + (defs.xcov_change_linear_max_factor - 1) * label[0] / defs.xmax) * 
        (1 + (defs.ycov_change_linear_max_factor - 1) * label[1] / defs.ymax) for label in labels]

def cov_change_skew(labels: np.ndarray, cov: np.ndarray) -> list[np.ndarray]:
    '''
    NOT IMPLEMENTED PROPERLY
    '''
    n_labels = len(labels)
    return [np.dot(cov, np.array([[np.cos(i * np.pi/n_labels), np.sin(i * np.pi/n_labels)], [-np.sin(i * np.pi/n_labels), np.cos(i * np.pi/n_labels)]])) for i in range(n_labels)]
