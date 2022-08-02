'''

2D-Gaussian Simulation

'''

import argparse
import datetime
from json import dump
from matplotlib import pyplot as plt
import numpy as np
import os
import re
import sys
import timeit
import torch
import torch.backends.cudnn as cudnn

import defs
from models.CCGAN import discriminator
from models.CCGAN import generator
from train import train_net
import train_utils

if __name__ == "__main__":

    #--------------------------------
    # define run name beforehand with -n flag
    # FILL

    # input a simple message if desired with -m flag
    note = ""
    if len(sys.argv) == 2 and sys.argv[1] == '-m':
        note = input("Add a note for this run: ")

    #--------------------------------------------------------------------------
    # System properties
    NGPU = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NCPU = 8

    #--------------------------------------------------------------------------
    # Torch/Cuda settings
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False

    #--------------------------------------------------------------------------
    # Extensibility Hooks
    # How to calculate labels and gaus peak points given the geometry of the
    # problem (e.g. circle vs line)

    if (defs.geo == "circle"):
        train_labels = train_utils.train_labels_circle
        normalize_labels = train_utils.normalize_labels_circle
        gaus_point = lambda labels: train_utils.gaus_point_circle(labels, defs.val)
        plot_lims = lambda: train_utils.plot_lims_circle(radius=defs.val)
    elif (defs.geo == "line"):
        train_labels = train_utils.train_labels_line_1d
        normalize_labels = train_utils.normalize_labels_line_1d
        gaus_point = lambda labels: train_utils.gaus_point_line_1d(labels, defs.val)
        plot_lims = train_utils.plot_lims_line_1d() 

    cov_mtx = lambda labels: train_utils.cov_change_const(labels, train_utils.cov_xy(defs.sigma_gaus))

    # dimensions of samples
    n_features = 2 # 2-D is this just dim_gan?? need to understand what that is

    # labels for training
    labels_train = train_labels(defs.ngaus)

    ### threshold to determine high quality samples
    quality_threshold = defs.sigma_gaus*4 #good samples are within 4 standard deviation

    #------------------------------------------------------------------------------
    # GAN definitions
    gen = generator(ngpu=NGPU, nz=defs.dim_gan, out_dim=n_features, val=defs.val, geo=defs.geo)
    dis = discriminator(ngpu=NGPU, input_dim=n_features, val=defs.val, geo=defs.geo)
    nlinear_g = len([m for m in gen.modules() if isinstance(m, torch.nn.Linear)])
    nlinear_d = len([m for m in dis.modules() if isinstance(m, torch.nn.Linear)])

    #-------------------------------
    # Output folders
    runs = np.empty((0,), dtype=int)
    p = re.compile(r"run_(\d+)")
    for d in os.listdir(defs.root_path + '/output/'):
        m = p.match(d)
        if m:
            runs = np.append(runs, int(m.group(1)))

    if len(runs) == 0:
        current_run_dir = defs.root_path + '/output/run_0/'
    else:
        all = np.linspace(0, np.max(runs) + 1, np.max(runs) + 2, dtype=int)
        diff = np.setdiff1d(all, runs)
        current_run_dir = defs.root_path + '/output/run_%i/'%(np.min(diff))

    save_models_dir = current_run_dir + 'saved_models/'
    os.makedirs(save_models_dir,exist_ok=True)
    save_data_dir = current_run_dir + 'saved_data/'
    os.makedirs(save_data_dir,exist_ok=True)

    #--------------------------------------------------------------------------
    # Store run properties (network and geometry)
    dict_params = {
        "note": note,
        "date": str(datetime.datetime.now()),
        "run_dir": current_run_dir,
        "seed": defs.seed,
        "nsim": defs.nsim,
        "niter": defs.niter,
        "niter_resume": defs.niter_resume,
        "niter_save": defs.niter_save,
        "nlinear_d": nlinear_d,
        "nlinear_g": nlinear_g,
        "batch_size_disc": defs.nbatch_d,
        "batch_size_gene": defs.nbatch_g,
        "sigma_kernel": defs.sigma_kernel,
        "kappa": defs.kappa,
        "dim_gan": defs.dim_gan,
        "lr": defs.lr,
        "geo": defs.geo,
        "threshold_type": defs.thresh,
        "soft_weight_threshold": defs.soft_weight_thresh,
        
        "n_gaussians": defs.ngaus,
        "n_gaussians_plot": defs.ngausplot,
        "n_samples_train": defs.nsamp,
        "sigma_gaussian": defs.sigma_gaus,
        "val": defs.val,
        "xmin": defs.xmin,
        "xmax": defs.xmax,
        "xbins": defs.xbins,
        "ymin": defs.ymin,
        "ymax": defs.ymax,
        "ybins": defs.ybins,
        "xcov_change_linear_max_factor": defs.xcov_change_linear_max_factor,
        "ycov_change_linear_max_factor": defs.ycov_change_linear_max_factor,
    }

    with open(current_run_dir + "run_parameters.json", 'w+') as filename_params_json:
        dump(dict_params, filename_params_json, indent=4)

    ###########################################################################
    '''                           Start Experiment                          '''
    ###########################################################################

    log = current_run_dir + "log.txt"
    log_file = open(log, 'w+')

    print("==================================================================================================")
    print("\nBegin The Experiment; Start Training (geo: {})>>>".format(defs.geo))
    print("==================================================================================================", file=log_file)
    print("\nBegin The Experiment; Start Training (geo: {})>>>".format(defs.geo), file=log_file)
    log_file.close()

    start = timeit.default_timer()
    for sim in range(defs.nsim):

        log_file = open(log, 'a+')

        print("\nSimulation %i" % (sim))
        print("\nSimulation %i" % (sim), file=log_file)

        #----------------------------------------------------------------------
        # Real data generation
                
        # List of individual data points/labels being trained with
        gaus_points_train = gaus_point(labels_train)
        
        # Covariance matrix for each point sampled
        cov_mtxs_train = cov_mtx(gaus_points_train)

        # Samples from gaussian
        samples_train, sampled_labels_train = train_utils.sample_real_gaussian(
            defs.nsamp, labels_train, gaus_points_train, cov_mtxs_train)

        filename_samples_npy = save_data_dir + 'samples_train_' + str(sim) + '.npy'
        filename_labels_npy = save_data_dir + 'labels_train_' + str(sim) + '.npy'

        np.save(filename_samples_npy, samples_train, allow_pickle=False)
        np.save(filename_labels_npy, sampled_labels_train, allow_pickle=False)

        # Preprocessing on labels
        sampled_labels_train_norm = normalize_labels(sampled_labels_train)

        # rule-of-thumb for the bandwidth selection
        if defs.sigma_kernel<0:
            std_labels_train_norm = np.std(sampled_labels_train_norm)
            defs.sigma_kernel = 1.06*std_labels_train_norm*(len(sampled_labels_train_norm))**(-1/5)

            print("Use rule-of-thumb formula to compute kernel_sigma >>>")
            print("Use rule-of-thumb formula to compute kernel_sigma >>>", file=log_file)

        # Determine vicinity parameter kappa/nu
        if defs.kappa < 0:
            kappa_base = np.abs(defs.kappa)/defs.ngaus

            if defs.thresh == "hard":
                defs.kappa = kappa_base
            else:
                defs.kappa = 1/kappa_base**2

        #----------------------------------------------------------------------
        # Train the CCGAN

        print(("{}/{}, {}, Sigma is {:04f}, Kappa is {:04f}".format(sim+1, defs.nsim, defs.thresh, defs.sigma_kernel, defs.kappa)))
        print(("{}/{}, {}, Sigma is {:04f}, Kappa is {:04f}".format(sim+1, defs.nsim, defs.thresh, defs.sigma_kernel, defs.kappa)), file=log_file)
        log_file.close()

        # CCGAN model
        filename_gan = save_models_dir + '/CCGAN_niter_{}_sim_{}.pth'.format(defs.niter, sim)

        # Training samples with labels array
        train_data = np.concatenate((np.array([sampled_labels_train_norm]).T, samples_train), axis=1)
        # Unique labels
        uniques = np.unique(train_data[:, 0], return_index=True, return_counts=True)

        # Train!
        gen, dis = train_net(gen, dis, defs.sigma_kernel, defs.kappa, train_data, uniques, save_models_dir=save_models_dir, log=log)

        # Store model
        torch.save({
            'dis_state_dict': dis.state_dict(),
            'gen_state_dict': gen.state_dict()
        }, filename_gan)

        for name, module in dis.named_children():
            print('resetting', name)
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

        for name, module in gen.named_children():
            print('resetting', name)
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

        # Increment seed for next simulation
        defs.seed += 1
            
    stop = timeit.default_timer()
    log_file = open(log, 'a+')
    print("GAN training finished; Time elapsed: {:04f}s".format(stop - start))
    print("\n{}, Sigma is {:04f}, Kappa is {:04f}".format(defs.thresh, defs.sigma_kernel, defs.kappa))
    print("\n===================================================================================================")
    print("GAN training finished; Time elapsed: {:04f}s".format(stop - start), file=log_file)
    print("\n{}, Sigma is {:04f}, Kappa is {:04f}".format(defs.thresh, defs.sigma_kernel, defs.kappa), file=log_file)
    print("\n===================================================================================================", file=log_file)
    log_file.close()