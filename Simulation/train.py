"""
Train a regression DCGAN

"""

# from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from numba import njit
import torch
from typing import Tuple
import numpy as np
import os
import timeit

from models.CCGAN import discriminator, generator
import defs

''' Settings '''
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# some parameters in opts
niter = defs.niter
dim_gan = defs.dim_gan
lr_g = defs.lr
lr_d = defs.lr
save_niter_freq = defs.niter_save
batch_size_disc = defs.nbatch_d
batch_size_gene = defs.nbatch_g

threshold_type = defs.thresh
nonzero_soft_weight_threshold = defs.soft_weight_thresh
n_samples = defs.nsamp
# rng = torch.manual_seed(defs.seed)
rng = np.random.default_rng(defs.seed)


@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

def train_CCGAN(gen, dis, sigma_kernel, kappa, train_data, uniques, save_models_dir=None, log=None) -> Tuple[generator, discriminator]:

    times = np.zeros((15,))

    if log is not None:
        log_file = open(log, 'a+')
    else:
        log_file = None

    gen = gen.to(device)
    dis = dis.to(device)

    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_dis = torch.optim.Adam(dis.parameters(), lr=lr_d, betas=(0.5, 0.999))

    if save_models_dir is not None and defs.niter_resume>0:
        save_file = save_models_dir + "/CcGAN_checkpoint_intrain/CcGAN_checkpoint_niter_{}.pth".format(defs.niter_resumes)
        checkpoint = torch.load(save_file)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        dis.load_state_dict(checkpoint['dis_state_dict'])
        optimizer_gen.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizer_dis.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])

    unique_train_labels = uniques[0]

    start_time = timeit.default_timer()

    for iter in range(defs.niter_resume, niter):

        iter_start_time = timeit.default_timer()

        '''  Train Discriminator   '''
        ## randomly draw batch_size_disc y's from unique_train_labels
        batch_target_labels_raw = unique_train_labels[rng.integers(len(unique_train_labels), size=batch_size_disc)]
        ## add Gaussian noise; we estimate image distribution conditional on these labels
        batch_epsilons = rng.normal(0, sigma_kernel, batch_size_disc)
        batch_target_labels = batch_target_labels_raw + batch_epsilons

        ## only for similation - THESE CAUSED EDGE CASE ISSUES
        # batch_target_labels[batch_target_labels<0] = batch_target_labels[batch_target_labels<0] + 1
        # batch_target_labels[batch_target_labels>1] = batch_target_labels[batch_target_labels>1] - 1

        ## find index of real images with labels in the vicinity of batch_target_labels
        ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
        batch_real_indx = np.ndarray((batch_size_disc, 3), dtype=float) #index of images in the datata; the labels of these images are in the vicinity

        batch_start_time = timeit.default_timer()

        ## prepare discriminator batch
        for j in range(batch_size_disc):

            vicinity_start_time = timeit.default_timer()

            if threshold_type == "hard":
                indices = np.where(np.abs(unique_train_labels-batch_target_labels[j])<= kappa)[0]
            else:
                # reverse the weight function for SVDL
                indices = np.where((unique_train_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]

            ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
            
            while len(indices)<1:
                epsilon_j = rng.normal(0, sigma_kernel, 1)
                batch_target_labels[j] = batch_target_labels_raw[j] + epsilon_j
                # ## only for similation - BAD
                # if batch_target_labels[j]<0:
                #     batch_target_labels[j] = batch_target_labels[j] + 1
                # if batch_target_labels[j]>1:
                #     batch_target_labels[j] = batch_target_labels[j] - 1
                # index for real images
                if threshold_type == "hard":
                    indices = np.where(np.abs(unique_train_labels-batch_target_labels[j])<= kappa)[0]
                else:
                    # reverse the weight function for SVDL
                    indices = np.where((unique_train_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]
                
            vicinity_end_time = timeit.default_timer()
            vicinity_time = vicinity_end_time - vicinity_start_time
            times[9] += vicinity_time

            labelling_start_time = timeit.default_timer()

            uniidx = uniques[1][indices]
            unicts = uniques[2][indices]

            # print(indices)
            # print(uniidx)
            # print(unicts)

            rng_choice_start_time = timeit.default_timer()
            near_cts_cum = np.cumsum(unicts)
            # print(near_cts_cum)
            choice = rng.integers(near_cts_cum[-1])
            # print(choice)
            # idx_idx = np.nonzero(near_cts_cum > rng.integers(near_cts_cum[-1]))[0][0] #argmax, where, or nonzero - nonzero is fastest? try numba or f2py
            idx_idx = index(near_cts_cum > choice, True)[0]
            # print(idx_idx)



            # near_idx = uniidx[indices]
            # near_cts = unicts[indices]
            # idx_idx = rng.choice(indices, p=near_cts/np.sum(near_cts))
            
            rng_choice_end_time = timeit.default_timer()
            rng_choice_time = rng_choice_end_time - rng_choice_start_time
            times[14] += rng_choice_time

            which_cts = unicts[idx_idx]
            which_idx = uniidx[idx_idx]
            # print(which_idx)
            # sample_idx = rng.integers(which_cts)
            sample_idx = uniidx[0] + choice
            # print(sample_idx)
            # which_row = train_data[which_idx][sample_idx]
            which_row = train_data[sample_idx]
            # print(which_row)

            labelling_end_time = timeit.default_timer()

            labelling_time = labelling_end_time - labelling_start_time
            times[11] += labelling_time
            
            batch_real_indx[j] = which_row

            # exit()


        # make the math ops all 1
        bounds_start_time = timeit.default_timer()
        ## labels for fake images generation
        if threshold_type == "hard":
            lb = batch_target_labels - kappa
            ub = batch_target_labels + kappa
        else:
            lb = batch_target_labels - np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
            ub = batch_target_labels + np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)

        mins = np.apply_along_axis(lambda x: np.maximum(0.0, x), axis=0, arr=lb)
        maxs = np.apply_along_axis(lambda x: np.minimum(x, 1.0), axis=0, arr=ub)
        bounds = np.concatenate(([mins], [maxs]), axis=0).T
        batch_fake_labels = np.apply_along_axis(lambda x: rng.uniform(x[0], x[1]), axis=1, arr=bounds)

        bounds_end_time = timeit.default_timer()
        bounds_time = bounds_end_time - bounds_start_time
        times[12] += bounds_time

        batch_end_time = timeit.default_timer()

        ## draw the real image batch from the training set
        batch_real_samples = batch_real_indx[:, 1:]
        batch_real_labels = batch_real_indx[:, 0]
        batch_real_samples = torch.from_numpy(batch_real_samples).type(torch.float).to(device)
        batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).to(device)

        ## generate the fake image batch
        batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).to(device)
        z = torch.from_numpy(rng.normal(size=(batch_size_disc, dim_gan))).type(torch.float).to(device)
        
        gen_dis_start_time = timeit.default_timer()
        batch_fake_samples = gen(z, batch_fake_labels)
        gen_dis_end_time = timeit.default_timer()

        ## target labels on gpu
        batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

        ## weight vector
        if threshold_type == "soft":
            real_weights = torch.exp(-kappa*(batch_real_labels-batch_target_labels)**2).to(device)
            fake_weights = torch.exp(-kappa*(batch_fake_labels-batch_target_labels)**2).to(device)
        else:
            real_weights = torch.ones(batch_size_disc, dtype=torch.float).to(device)
            fake_weights = torch.ones(batch_size_disc, dtype=torch.float).to(device)

        # forward pass
        dis_dis_start_time = timeit.default_timer()
        real_dis_out = dis(batch_real_samples, batch_target_labels)
        fake_dis_out = dis(batch_fake_samples.detach(), batch_target_labels)
        dis_dis_end_time = timeit.default_timer()

        d_loss = - torch.mean(real_weights.reshape(-1) * torch.log(real_dis_out.reshape(-1)+1e-20)) - torch.mean(fake_weights.reshape(-1) * torch.log(1 - fake_dis_out.reshape(-1)+1e-20))

        dis_opt_start_time = timeit.default_timer()
        optimizer_dis.zero_grad()
        d_loss.backward()
        optimizer_dis.step()

        dis_train_end_time = timeit.default_timer()

        '''  Train Generator   '''
        gen.train()

        # generate fake images
        ## randomly draw batch_size_disc y's from unique_train_labels
        batch_target_labels_raw = rng.choice(unique_train_labels, size=batch_size_gene, replace=True)
        ## add Gaussian noise; we estimate image distribution conditional on these labels
        batch_epsilons = rng.normal(0, sigma_kernel, batch_size_gene)
        batch_target_labels = batch_target_labels_raw + batch_epsilons
        # if labels are cyclic, cycle labels to be between 0. and 1.
        # batch_target_labels[batch_target_labels<0] = batch_target_labels[batch_target_labels<0] + 1
        # batch_target_labels[batch_target_labels>1] = batch_target_labels[batch_target_labels>1] - 1
        # set out-of-bounds labels to be instead on the boundary
        # batch_target_labels[batch_target_labels<0] = 0
        # batch_target_labels[batch_target_labels>1] = 1
        # constrain labels between 0. and 1. not inclusive
        
        # #fix out-of-bounds raw labels
        # batch_labels_oob_init_indx = np.where(np.abs(batch_target_labels - 0.5) >= 0.5)
        # batch_labels_oob = batch_target_labels_raw[batch_labels_oob_init_indx]
        # while len(batch_labels_oob) != 0:
        #     print("re-nudging oob raw labels")
        #     batch_epsilons_oob = rng.normal(0, sigma_kernel, len(batch_labels_oob))
        #     batch_labels_oob = batch_target_labels_raw[batch_labels_oob_init_indx] + batch_epsilons_oob
        #     batch_labels_oob_indx = batch_labels_oob_init_indx[np.where(np.abs(batch_labels_oob - 0.5) >= 0.5)]
        #     batch_labels_inb_indx = np.setdiff1d(batch_labels_oob_init_indx, batch_labels_oob_indx)

        #     batch_target_labels[batch_labels_inb_indx] = batch_labels_oob[np.abs(batch_labels_oob - 0.5) < 0.5]
        
        # batch_target_labels = batch_target_labels[np.abs(batch_target_labels - 0.5) < 0.5]
        batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

        z = torch.from_numpy(rng.normal(size=(batch_size_gene, dim_gan))).type(torch.float).to(device)

        gen_gen_start_time = timeit.default_timer()
        batch_fake_samples = gen(z, batch_target_labels)
        gen_gen_end_time = timeit.default_timer()

        # loss
        dis_gen_start_time = timeit.default_timer()
        dis_out = dis(batch_fake_samples, batch_target_labels)
        dis_gen_end_time = timeit.default_timer()
        g_loss = - torch.mean(torch.log(dis_out+1e-20))

        # backward
        gen_opt_start_time = timeit.default_timer()
        optimizer_gen.zero_grad()
        g_loss.backward()
        optimizer_gen.step()
        gen_train_end_time = timeit.default_timer()

        # print loss
        if iter%100 == 0:
            print("CcGAN: [Iter %d/%d] [D loss: %.4e] [G loss: %.4e] [real prob: %.3f] [fake prob: %.3f] [Time: %.4fs]"%(iter+1, niter, d_loss.item(), g_loss.item(), real_dis_out.mean().item(), fake_dis_out.mean().item(), timeit.default_timer()-start_time))
            if log_file is not None:
                print("CcGAN: [Iter %d/%d] [D loss: %.4e] [G loss: %.4e] [real prob: %.3f] [fake prob: %.3f] [Time: %.4fs]"%(iter+1, niter, d_loss.item(), g_loss.item(), real_dis_out.mean().item(), fake_dis_out.mean().item(), timeit.default_timer()-start_time), file=log_file)

        if save_models_dir is not None and ((iter+1) % save_niter_freq == 0 or (iter+1) == niter):
            save_file = save_models_dir + "/checkpoints/CCGAN_iter_{}.pth".format(iter+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'gen_state_dict': gen.state_dict(),
                    'dis_state_dict': dis.state_dict(),
                    'optimizerG_state_dict': optimizer_gen.state_dict(),
                    'optimizerD_state_dict': optimizer_dis.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)

        iter_end_time = timeit.default_timer()

        
        pre_batch_time = batch_start_time - iter_start_time
        batch_time = batch_end_time - batch_start_time
        post_batch_time = dis_train_end_time - batch_end_time
        gen_train_time = iter_end_time - dis_train_end_time
        dis_time = dis_dis_end_time - dis_dis_start_time + dis_gen_end_time - dis_gen_start_time
        gen_time = gen_dis_end_time - gen_dis_start_time + gen_gen_end_time - gen_gen_start_time
        dis_opt_time = dis_train_end_time - dis_opt_start_time
        gen_opt_time = gen_train_end_time - gen_opt_start_time
        total_time = iter_end_time - iter_start_time

        times[0] += pre_batch_time
        times[1] += batch_time
        times[2] += post_batch_time
        times[3] += gen_train_time
        times[4] += dis_time
        times[5] += gen_time
        times[6] += dis_opt_time
        times[7] += gen_opt_time
        times[8] += total_time

    log_file.close()

    return gen, dis, times