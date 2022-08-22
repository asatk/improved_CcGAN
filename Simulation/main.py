'''

2D-Gaussian Simulation

'''

import argparse
import datetime
from json import dump
import numpy as np
import os
import re
import timeit
import torch
import torch.backends.cudnn as cudnn

from Run import Run
import defs
from models.CCGAN import discriminator
from models.CCGAN import generator
from train import train_net
import train_utils

def main(run: Run):

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

    if (run.geo == "circle"):
        train_labels = train_utils.train_labels_circle
        normalize_labels = train_utils.normalize_labels_circle
        gaus_point = lambda labels: train_utils.gaus_point_circle(labels, run.val)
    elif (run.geo == "line"):
        train_labels = train_utils.train_labels_line_1d
        normalize_labels = train_utils.normalize_labels_line_1d
        gaus_point = lambda labels: train_utils.gaus_point_line_1d(labels, run.val)

    cov_mtx = lambda labels: train_utils.cov_change_const(labels, train_utils.cov_xy(run.sigma_gaus))
    # cov_mtx = lambda labels: train_utils.cov_change_radial(labels, train_utils.cov_xy(run.sigma_gaus))

    # dimensions of samples
    n_features = 2 # 2-D is this just dim_gan?? need to understand what that is

    # labels for training
    labels_train = train_labels(run.ngaus)

    #------------------------------------------------------------------------------
    # GAN definitions
    gen = generator(ngpu=NGPU, nz=run.dim_gan, out_dim=n_features, val=run.val, geo=run.geo)
    dis = discriminator(ngpu=NGPU, input_dim=n_features, val=run.val, geo=run.geo)
    nlinear_g = len([m for m in gen.modules() if isinstance(m, torch.nn.Linear)])
    nlinear_d = len([m for m in dis.modules() if isinstance(m, torch.nn.Linear)])

    #-------------------------------
    # Output folders
    runs = np.empty((0,), dtype=int)
    p = re.compile(r"run_(\d+)")
    for d in os.listdir(run.root_path + '/output/'):
        m = p.match(d)
        if m:
            runs = np.append(runs, int(m.group(1)))

    if run.name != "":
        current_run_dir = run.root_path + '/output/%s/'%(run.name)
    else:
        if len(runs) == 0:
            current_run_dir = run.root_path + '/output/run_0/'
        else:
            nums = np.linspace(0, np.max(runs) + 1, np.max(runs) + 2, dtype=int)
            diff = np.setdiff1d(nums, runs)
            current_run_dir = run.root_path + '/output/run_%i/'%(np.min(diff))

    save_models_dir = current_run_dir + 'saved_models/'
    os.makedirs(save_models_dir,exist_ok=True)
    save_data_dir = current_run_dir + 'saved_data/'
    os.makedirs(save_data_dir,exist_ok=True)

    #--------------------------------------------------------------------------
    # Store run properties (network and geometry)
    dict_params = {
        "memo": run.memo,
        "date": str(datetime.datetime.now()),
        "run_dir": current_run_dir,
        "seed": run.seed,
        "nsim": run.nsim,
        "niter": run.niter,
        "niter_resume": run.niter_resume,
        "niter_save": run.niter_save,
        "nlinear_d": nlinear_d,
        "nlinear_g": nlinear_g,
        "batch_size_disc": run.nbatch_d,
        "batch_size_gene": run.nbatch_g,
        "sigma_kernel": run.sigma_kernel,
        "kappa": run.kappa,
        "dim_gan": run.dim_gan,
        "lr": run.lr,
        "geo": run.geo,
        "threshold_type": run.thresh,
        "soft_weight_threshold": run.soft_weight_thresh,
        
        "n_gaussians": run.ngaus,
        "n_gaussians_plot": run.ngausplot,
        "n_samples_train": run.nsamp,
        "sigma_gaussian": run.sigma_gaus,
        "val": run.val,
        "xmin": run.xmin,
        "xmax": run.xmax,
        "xbins": run.xbins,
        "ymin": run.ymin,
        "ymax": run.ymax,
        "ybins": run.ybins,
        "xcov_change_linear_max_factor": run.xcov_change_linear_max_factor,
        "ycov_change_linear_max_factor": run.ycov_change_linear_max_factor,
    }

    with open(current_run_dir + "run_parameters.json", 'w+') as filename_params_json:
        dump(dict_params, filename_params_json, indent=4)

    ###########################################################################
    '''                           Start Experiment                          '''
    ###########################################################################

    log = current_run_dir + "log.txt"
    log_file = open(log, 'w+')

    print("==================================================================================================")
    print("\nBegin The Experiment; Start Training (geo: {})>>>".format(run.geo))
    print("==================================================================================================", file=log_file)
    print("\nBegin The Experiment; Start Training (geo: {})>>>".format(run.geo), file=log_file)
    log_file.close()

    start = timeit.default_timer()
    for sim in range(run.nsim):

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
            run.nsamp, labels_train, gaus_points_train, cov_mtxs_train)

        filename_samples_npy = save_data_dir + 'samples_train_' + str(sim) + '.npy'
        filename_labels_npy = save_data_dir + 'labels_train_' + str(sim) + '.npy'

        np.save(filename_samples_npy, samples_train, allow_pickle=False)
        np.save(filename_labels_npy, sampled_labels_train, allow_pickle=False)

        # Preprocessing on labels
        sampled_labels_train_norm = normalize_labels(sampled_labels_train)

        # rule-of-thumb for the bandwidth selection
        if run.sigma_kernel<0:
            std_labels_train_norm = np.std(sampled_labels_train_norm)
            run.sigma_kernel = 1.06*std_labels_train_norm*(len(sampled_labels_train_norm))**(-1/5)

            print("Use rule-of-thumb formula to compute kernel_sigma >>>")
            print("Use rule-of-thumb formula to compute kernel_sigma >>>", file=log_file)

        # Determine vicinity parameter kappa/nu
        if run.kappa < 0:
            kappa_base = np.abs(run.kappa)/run.ngaus

            if run.thresh == "hard":
                run.kappa = kappa_base
            else:
                run.kappa = 1/kappa_base**2

        #----------------------------------------------------------------------
        # Train the CCGAN

        print(("{}/{}, {}, Sigma is {:04f}, Kappa is {:04f}".format(sim+1, run.nsim, run.thresh, run.sigma_kernel, run.kappa)))
        print(("{}/{}, {}, Sigma is {:04f}, Kappa is {:04f}".format(sim+1, run.nsim, run.thresh, run.sigma_kernel, run.kappa)), file=log_file)
        log_file.close()

        # CCGAN model
        filename_gan = save_models_dir + '/CCGAN_niter_{}_sim_{}.pth'.format(run.niter, sim)

        # Training samples with labels array
        train_data = np.concatenate((np.array([sampled_labels_train_norm]).T, samples_train), axis=1)
        # Unique labels
        uniques = np.unique(train_data[:, 0], return_index=True, return_counts=True)

        # Train!
        gen, dis = train_net(gen, dis, run.sigma_kernel, run.kappa, train_data, uniques, save_models_dir=save_models_dir, log=log)

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
        run.seed += 1
            
    stop = timeit.default_timer()
    log_file = open(log, 'a+')
    print("GAN training finished; Time elapsed: {:04f}s".format(stop - start))
    print("\n{}, Sigma is {:04f}, Kappa is {:04f}".format(run.thresh, run.sigma_kernel, run.kappa))
    print("\n===================================================================================================")
    print("GAN training finished; Time elapsed: {:04f}s".format(stop - start), file=log_file)
    print("\n{}, Sigma is {:04f}, Kappa is {:04f}".format(run.thresh, run.sigma_kernel, run.kappa), file=log_file)
    print("\n===================================================================================================", file=log_file)
    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--child", action="store_true", help="is main.py being run as a child process")
    parser.add_argument("-n", "--name", default="", help="name of this run")
    parser.add_argument("-m", "--memo", default="", help="note for this run")
    args = parser.parse_args()
    run = Run(dict((k, v) for k, v in defs.__dict__.items() if k[0] != '_'), name=args.name, memo=args.memo)
    main(run)