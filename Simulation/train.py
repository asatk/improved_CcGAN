"""
Train a regression DCGAN

"""

import torch
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

rng = np.random.default_rng(defs.seed)

def train_net(gen: generator, dis: discriminator, sigma_kernel: float, kappa: float, train_data: np.ndarray, uniques: list[np.ndarray], save_models_dir: str=None, log: str=None) -> tuple[generator, discriminator]:

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

        '''  Train Discriminator   '''
        ## randomly draw batch_size_disc y's from unique_train_labels
        batch_target_labels_raw = unique_train_labels[rng.integers(len(unique_train_labels), size=batch_size_disc)]
        ## add Gaussian noise; we estimate image distribution conditional on these labels
        batch_epsilons = rng.normal(0, sigma_kernel, batch_size_disc)
        batch_target_labels = batch_target_labels_raw + batch_epsilons

        # push epsilon inside?
        oob_labels = np.abs(batch_target_labels - 0.5) > 0.5

        if len(oob_labels) != 0:
            print("pushing epsilons inside: ", batch_target_labels[oob_labels], file=log_file)
            batch_target_labels[oob_labels] = batch_target_labels_raw[oob_labels] - batch_epsilons[oob_labels]

        ## find real samples with labels in the vicinity of batch_target_labels
        ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
        batch_real_choices = np.ndarray((batch_size_disc, 3), dtype=float) #choices of samples with their labels in the vicinity

        ## prepare discriminator batch
        for j in range(batch_size_disc):

            if threshold_type == "hard":
                indices = np.where(np.abs(unique_train_labels-batch_target_labels[j])<= kappa)[0]
            else:
                # reverse the weight function for SVDL
                indices = np.where((unique_train_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]

            ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
            while len(indices)<1:
                epsilon_j = rng.normal(0, sigma_kernel, 1)
                batch_target_labels[j] = batch_target_labels_raw[j] + epsilon_j
                
                # push epsilon inside?
                if np.abs(batch_target_labels[j] - 0.5) > 0.5:
                    print("pushing epsilon inside: ", batch_target_labels[j], file=log_file)
                    batch_target_labels[j] = batch_target_labels_raw - epsilon_j

                # index for real images
                if threshold_type == "hard":
                    indices = np.where(np.abs(unique_train_labels-batch_target_labels[j])<= kappa)[0]
                else:
                    # reverse the weight function for SVDL
                    indices = np.where((unique_train_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]

            near_cts_cum = np.cumsum(uniques[2][indices])
            choice = rng.integers(near_cts_cum[-1])
            
            sample_idx = uniques[1][indices][0] + choice
            batch_real_choices[j] = train_data[sample_idx]

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

        ## draw the real image batch from the training set
        batch_real_samples = batch_real_choices[:, 1:]
        batch_real_labels = batch_real_choices[:, 0]
        batch_real_samples = torch.from_numpy(batch_real_samples).type(torch.float).to(device)
        batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).to(device)

        ## generate the fake image batch
        batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).to(device)
        z = torch.from_numpy(rng.normal(size=(batch_size_disc, dim_gan))).type(torch.float).to(device)
        batch_fake_samples = gen(z, batch_fake_labels)

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
        real_dis_out = dis(batch_real_samples, batch_target_labels)
        fake_dis_out = dis(batch_fake_samples.detach(), batch_target_labels)

        d_loss = - torch.mean(real_weights.reshape(-1) * torch.log(real_dis_out.reshape(-1)+1e-20)) - torch.mean(fake_weights.reshape(-1) * torch.log(1 - fake_dis_out.reshape(-1)+1e-20))

        optimizer_dis.zero_grad()
        d_loss.backward()
        optimizer_dis.step()

        '''  Train Generator   '''
        gen.train()

        # generate fake images
        ## randomly draw batch_size_disc y's from unique_train_labels
        batch_target_labels_raw = rng.choice(unique_train_labels, size=batch_size_gene, replace=True)
        ## add Gaussian noise; we estimate image distribution conditional on these labels
        batch_epsilons = rng.normal(0, sigma_kernel, batch_size_gene)
        batch_target_labels = batch_target_labels_raw + batch_epsilons

        # push epsilon inside?
        oob_labels = np.abs(batch_target_labels - 0.5) > 0.5
        if len(oob_labels) != 0:
            print("pushing epsilons inside: ", batch_target_labels[oob_labels], file=log_file)
            batch_target_labels[oob_labels] = batch_target_labels_raw[oob_labels] - batch_epsilons[oob_labels]
        
        # batch_target_labels = batch_target_labels[np.abs(batch_target_labels - 0.5) < 0.5]
        batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

        z = torch.from_numpy(rng.normal(size=(batch_size_gene, dim_gan))).type(torch.float).to(device)
        batch_fake_samples = gen(z, batch_target_labels)

        # loss
        dis_out = dis(batch_fake_samples, batch_target_labels)
        g_loss = - torch.mean(torch.log(dis_out+1e-20))

        # backward
        optimizer_gen.zero_grad()
        g_loss.backward()
        optimizer_gen.step()

        # print loss
        if iter%100 == 0:
            print_time = timeit.default_timer()
            print("CcGAN: [Iter %d/%d] [D loss: %.4e] [G loss: %.4e] [real prob: %.3f] [fake prob: %.3f] [Time: %.4fs]"%(iter+1, niter, d_loss.item(), g_loss.item(), real_dis_out.mean().item(), fake_dis_out.mean().item(), print_time-start_time))
            if log_file is not None:
                print("CcGAN: [Iter %d/%d] [D loss: %.4e] [G loss: %.4e] [real prob: %.3f] [fake prob: %.3f] [Time: %.4fs]"%(iter+1, niter, d_loss.item(), g_loss.item(), real_dis_out.mean().item(), fake_dis_out.mean().item(), print_time-start_time), file=log_file)

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

    log_file.close()

    return gen, dis