'''

2D gaussian grid phi x omega ML Generation

'''
import numpy as np
import os
import subprocess
import json
import random
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from opts import parse_opts
# from PIL import Image
import timeit

from models import cont_cond_GAN as CCGAN

# from defs import PROJECT_DIR


# system
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCPU = 8

# seeds
seed = 100
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(seed)

# labels
# labels_phi_real = np.linspace(300,3000,num=10,endpoint=True)
# labels_phi_all = np.linspace(0,3000,num=1000+1,endpoint=True)
# labels_omega_real = np.linspace(0.2,2.0,num=10,endpoint=True)
# labels_omega_all = np.linspace(0,2.0,num=1000+1,endpoint=True)

# # load numpy histogram data
# gaus_grid = []
# for f in os.listdir(PROJECT_DIR+"/out/hist_npy/"):
#     arr_temp = np.load(PROJECT_DIR+"/out/hist_npy/"+f)
#     gaus_grid.append(arr_temp)

# load json hyperparameters
# hpars_path = PROJECT_DIR+"/src/hyperparams.json"
# with open(hpars_path) as hpars_json:
#     hpars = json.load(hpars_json)
#     vars = hpars['vars']
#     kappas = np.array(hpars['kappas'])
#     sigmas = np.array(hpars['sigmas'])

# learning parameters
args = parse_opts()
niters = args.niters_gan
resume_niters = args.resume_niters_gan
dim_gan = args.dim_gan
lr_g = args.lr_gan
lr_d = args.lr_gan
save_niters_freq = args.save_niters_freq
batch_size_D = args.batch_size_disc
batch_size_G = args.batch_size_gene

# n_vars = 2
n_vars = 1
# n_samples = len(os.listdir(PROJECT_DIR+"/out/hist_npy/"))
# n_samples = int(subprocess.check_output('ls -alFh ../out/hist_npy/ | grep "1800" | wc -l',shell=True))
# n_epochs = int(1e2)
# lr_g = 5e-5
# lr_d = 5e-5
# batch_size_D = 5
# batch_size_G = 128
use_hard_vicinity = True
soft_threshold = 1e-3
dim_latent_space = 128

# load data and labels in
# train_data = [0]*n_samples
# train_labels = np.zeros((n_samples,n_vars))

# second_label = 0.8e0

# associate each file's data with its label
# data_mapping_path = "data_mapping.json"
# with open(data_mapping_path) as json_file:
#     data_label_map = json.load(json_file)
#     # load each file from the data mapping
#     count = 0
#     for i,f in enumerate(data_label_map.keys()):
#         # train_labels[i] = data_label_map[f]
#         # train_data[i] = np.load(f)
#         temp_label = data_label_map[f]
#         if temp_label[1] == second_label:
#             train_labels[count] = float(temp_label[0])
#             train_data[count] = np.load(f)
#             count+=1
            

# train_data = np.array(train_data)

# for 1-D only
var = 0
# label = 1.8e3
# train_index = np.where(train_labels == 1.8e3)[0]
# train_labels = train_labels[train_index]
# train_data = train_data[train_index,:,:]

# output_shape = train_data[0].shape
# out_dim = np.prod(output_shape)
# print(output_shape)
# print(out_dim)

# print(train_index)
# print(train_labels)
# print(train_data)

noise_dim = 128

# gan
# netG = CCGAN.cont_cond_generator(nz=noise_dim,nlabels=1,out_dim=out_dim)
# netD = CCGAN.cont_cond_discriminator(input_dim=out_dim)
# netG.float()
# netD.float()

def train_CcGAN(kernel_sigma, kappa, train_samples, train_labels, netG, netD, save_images_folder, save_models_folder = None, plot_in_train=False, samples_tar_eval = None, angle_grid_eval = None, fig_size=5, point_size=None):
    #-----------TRAIN D-----------#
    netG.float()
    netD.float()

    sigmas = np.array([kernel_sigma])
    kappas = np.array([kappa])

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5,0.999))

    # generate a batch of target labels from all real labels
    batch_target_labels_raw = np.ndarray((batch_size_D,n_vars))
    for v in range(n_vars):
        batch_target_labels_raw[:,v] = np.random.choice(train_labels[:,v],size=batch_size_D,replace=True).transpose()

    # give each target label some noise
    batch_epsilons = np.ndarray((batch_size_D,n_vars))
    for v in range(n_vars):
        batch_epsilons[:,v] = np.random.normal(0,sigmas[v],batch_size_D).transpose()

    batch_target_labels = batch_target_labels_raw + batch_epsilons

    # find real data within bounds
    batch_real_index = np.zeros((batch_size_D),dtype=int)
    batch_fake_labels = np.zeros((batch_size_D,n_vars))

    def vicinity_hard(x,sample):
        return np.sum(np.square(np.divide(x-sample,kappas)))

    def vicinity_soft(x,sample):
        return np.exp(-1*vicinity_hard(x,sample))

    # iterate over the whole batch
    j=0
    reshuffle_count = 0
    while j < batch_size_D:
        # print(j)
        
        batch_target_label = batch_target_labels[j]
        # print("\n")
        # print("batch target label",batch_target_label)

        label_vicinity = np.ndarray(train_labels.shape)

        # iterate over every training label to find which are within the vicinity of the batch labels
        for i in range(len(train_labels)):
            train_label = train_labels[i]
            if use_hard_vicinity:
                hard = np.apply_along_axis(vicinity_hard,0,train_label,sample=batch_target_label)
                label_vicinity[i] = hard
            else:
                soft = np.apply_along_axis(vicinity_soft,0,train_label,sample=batch_target_label)
                label_vicinity[i] = soft
        
        if use_hard_vicinity:
            indices = np.unique(np.where(label_vicinity <= 1)[0])
        else:
            indices = np.unique(np.where(label_vicinity >= soft_threshold)[0])

        # reshuffle the batch target labels, redo that sample
        if len(indices) < 1:
            reshuffle_count += 1
            # print("RESHUFFLE COUNT",reshuffle_count)
            batch_epsilons_j = np.zeros((n_vars))
            for v in range(n_vars):
                batch_epsilons_j[v] = np.random.normal(0,sigmas[v],1).transpose()
            batch_target_labels[j] = batch_target_labels_raw[j] + batch_epsilons_j
            continue

        # print("RESHUFFLE COUNT",reshuffle_count)
        # print("BATCH SAMPLE",batch_target_label)

        # set the bounds for random draw of possible fake labels
        if use_hard_vicinity == "hard":
            lb = batch_target_labels[j] - kappas
            ub = batch_target_labels[j] + kappas
        else:
            lb = batch_target_labels[j] - np.sqrt(-1*np.log(soft_threshold)*kappas)
            ub = batch_target_labels[j] + np.sqrt(-1*np.log(soft_threshold)*kappas)

        # pick real sample in vicinity
        batch_real_index[j] = np.random.choice(indices,size=1)[0]

        # generate fake labels
        for v in range(n_vars):
            batch_fake_labels[j,v] = np.random.uniform(lb[v],ub[v],size=1)[0]
        
        j += 1
        reshuffle_count = 0

    batch_real_samples = train_data[batch_real_index]
    batch_real_labels = train_labels[batch_real_index]
    batch_real_samples = torch.from_numpy(batch_real_samples).type(torch.float).to(device)
    batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).to(device)

    print("BATCH REAL INDEX:\n",batch_real_index)
    print("BATCH REAL LABELS:\n",batch_real_labels)
    print("BATCH FAKE LABELS:\n",batch_fake_labels)
    print("BATCH TARGET LABELS:\n",batch_target_labels)

    batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).to(device)
    z = torch.randn(batch_size_D,dim_latent_space,dtype=torch.float).to(device)
    batch_fake_samples = netG(z, batch_fake_labels)

    batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

    if use_hard_vicinity:
        real_weights = torch.ones(batch_size_D, dtype=torch.float).to(device)
        fake_weights = torch.ones(batch_size_D, dtype=torch.float).to(device)
    else:
        real_weights = np.apply_along_axis(vicinity_hard,0,batch_real_labels,sample=batch_target_label)
        fake_weights = np.apply_along_axis(vicinity_hard,0,batch_fake_labels,sample=batch_target_label)

    real_dis_out = netD(batch_real_samples, batch_target_labels)
    fake_dis_out = netD(batch_fake_samples.detach(), batch_target_labels)

    d_loss = - torch.mean(real_weights.view(-1) * torch.log(real_dis_out.view(-1)+1e-20)) - \
            torch.mean(fake_weights.view(-1) * torch.log(1-fake_dis_out.view(-1)+1e-20))

    optimizerD.zero_grad()
    d_loss.backward()
    optimizerD.step()

    #-----------TRAIN G-----------#

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5,0.999))

    # generate a batch of target labels from all real labels
    batch_target_labels_raw = np.ndarray((batch_size_G,n_vars))
    for v in range(n_vars):
        batch_target_labels_raw[:,v] = np.random.choice(train_labels[:,v],size=batch_size_G,replace=True).transpose()

    # give each target label some noise
    batch_epsilons = np.ndarray((batch_size_G,n_vars))
    for v in range(n_vars):
        batch_epsilons[:,v] = np.random.normal(0,sigmas[v],batch_size_G).transpose()

    batch_target_labels = batch_target_labels_raw + batch_epsilons
    batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

    z = torch.randn(batch_size_G,noise_dim,dtype=torch.float).to(device)
    batch_fake_samples = netG(z,batch_target_labels)

    dis_out = netD(batch_fake_samples,batch_target_labels)
    g_loss = - torch.mean(torch.log(dis_out+1e-20))

    optimizerG.zero_grad()
    g_loss.backward()
    optimizerG.step()

    return netG, netD

def SampCcGAN_given_label(netG, label, path=None, NFAKE = 10000, batch_size = 500, num_features=2):
    '''
    label: normalized label in [0,1]
    '''
    if batch_size>NFAKE:
        batch_size = NFAKE
    fake_samples = np.zeros((NFAKE+batch_size, num_features), dtype=np.float)
    netG=netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)
            y = np.ones(batch_size) * label
            y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
            batch_fake_samples = netG(z, y)
            fake_samples[tmp:(tmp+batch_size)] = batch_fake_samples.cpu().detach().numpy()
            tmp += batch_size

    #remove extra entries
    fake_samples = fake_samples[0:NFAKE]
    fake_angles = np.ones(NFAKE) * label #use assigned label

    if path is not None:
        raw_fake_samples = (fake_samples*0.5+0.5)*255.0
        raw_fake_samples = raw_fake_samples.astype(np.uint8)
        for i in range(NFAKE):
            filename = path + '/' + str(i) + '.jpg'
            im = Image.fromarray(raw_fake_samples[i][0], mode='L')
            im = im.save(filename)

    return fake_samples, fake_angles

def loss_D():

    #-----------TRAIN D-----------#

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5,0.999))

    # generate a batch of target labels from all real labels
    batch_target_labels_raw = np.ndarray((batch_size_D,n_vars))
    for v in range(n_vars):
        batch_target_labels_raw[:,v] = np.random.choice(train_labels[:,v],size=batch_size_D,replace=True).transpose()

    # give each target label some noise
    batch_epsilons = np.ndarray((batch_size_D,n_vars))
    for v in range(n_vars):
        batch_epsilons[:,v] = np.random.normal(0,sigmas[v],batch_size_D).transpose()

    batch_target_labels = batch_target_labels_raw + batch_epsilons

    # find real data within bounds
    batch_real_index = np.zeros((batch_size_D),dtype=int)
    batch_fake_labels = np.zeros((batch_size_D,n_vars))

    def vicinity_hard(x,sample):
        return np.sum(np.square(np.divide(x-sample,kappas)))

    def vicinity_soft(x,sample):
        return np.exp(-1*vicinity_hard(x,sample))

    # iterate over the whole batch
    j=0
    reshuffle_count = 0
    while j < batch_size_D:
        # print(j)
        
        batch_target_label = batch_target_labels[j]
        # print("\n")
        # print("batch target label",batch_target_label)

        label_vicinity = np.ndarray(train_labels.shape)

        # iterate over every training label to find which are within the vicinity of the batch labels
        for i in range(len(train_labels)):
            train_label = train_labels[i]
            if use_hard_vicinity:
                hard = np.apply_along_axis(vicinity_hard,0,train_label,sample=batch_target_label)
                label_vicinity[i] = hard
            else:
                soft = np.apply_along_axis(vicinity_soft,0,train_label,sample=batch_target_label)
                label_vicinity[i] = soft
        
        if use_hard_vicinity:
            indices = np.unique(np.where(label_vicinity <= 1)[0])
        else:
            indices = np.unique(np.where(label_vicinity >= soft_threshold)[0])

        # reshuffle the batch target labels, redo that sample
        if len(indices) < 1:
            reshuffle_count += 1
            # print("RESHUFFLE COUNT",reshuffle_count)
            batch_epsilons_j = np.zeros((n_vars))
            for v in range(n_vars):
                batch_epsilons_j[v] = np.random.normal(0,sigmas[v],1).transpose()
            batch_target_labels[j] = batch_target_labels_raw[j] + batch_epsilons_j
            continue

        # print("RESHUFFLE COUNT",reshuffle_count)
        # print("BATCH SAMPLE",batch_target_label)

        # set the bounds for random draw of possible fake labels
        if use_hard_vicinity == "hard":
            lb = batch_target_labels[j] - kappas
            ub = batch_target_labels[j] + kappas
        else:
            lb = batch_target_labels[j] - np.sqrt(-1*np.log(soft_threshold)*kappas)
            ub = batch_target_labels[j] + np.sqrt(-1*np.log(soft_threshold)*kappas)

        # pick real sample in vicinity
        batch_real_index[j] = np.random.choice(indices,size=1)[0]

        # generate fake labels
        for v in range(n_vars):
            batch_fake_labels[j,v] = np.random.uniform(lb[v],ub[v],size=1)[0]
        
        j += 1
        reshuffle_count = 0

    batch_real_samples = train_data[batch_real_index]
    batch_real_labels = train_labels[batch_real_index]
    batch_real_samples = torch.from_numpy(batch_real_samples).type(torch.float).to(device)
    batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).to(device)

    print("BATCH REAL INDEX:\n",batch_real_index)
    print("BATCH REAL LABELS:\n",batch_real_labels)
    print("BATCH FAKE LABELS:\n",batch_fake_labels)
    print("BATCH TARGET LABELS:\n",batch_target_labels)

    batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).to(device)
    z = torch.randn(batch_size_D,dim_latent_space,dtype=torch.float).to(device)
    batch_fake_samples = netG(z, batch_fake_labels)

    batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

    if use_hard_vicinity:
        real_weights = torch.ones(batch_size_D, dtype=torch.float).to(device)
        fake_weights = torch.ones(batch_size_D, dtype=torch.float).to(device)
    else:
        real_weights = np.apply_along_axis(vicinity_hard,0,batch_real_labels,sample=batch_target_label)
        fake_weights = np.apply_along_axis(vicinity_hard,0,batch_fake_labels,sample=batch_target_label)

    real_dis_out = netD(batch_real_samples, batch_target_labels)
    fake_dis_out = netD(batch_fake_samples.detach(), batch_target_labels)

    d_loss = - torch.mean(real_weights.view(-1) * torch.log(real_dis_out.view(-1)+1e-20)) - \
            torch.mean(fake_weights.view(-1) * torch.log(1-fake_dis_out.view(-1)+1e-20))

    optimizerD.zero_grad()
    d_loss.backward()
    optimizerD.step()

    return netG

def loss_G():

    #-----------TRAIN G-----------#

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5,0.999))

    # generate a batch of target labels from all real labels
    batch_target_labels_raw = np.ndarray((batch_size_G,n_vars))
    for v in range(n_vars):
        batch_target_labels_raw[:,v] = np.random.choice(train_labels[:,v],size=batch_size_G,replace=True).transpose()

    # give each target label some noise
    batch_epsilons = np.ndarray((batch_size_G,n_vars))
    for v in range(n_vars):
        batch_epsilons[:,v] = np.random.normal(0,sigmas[v],batch_size_G).transpose()

    batch_target_labels = batch_target_labels_raw + batch_epsilons
    batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

    z = torch.randn(batch_size_G,noise_dim,dtype=torch.float).to(device)
    batch_fake_samples = netG(z,batch_target_labels)

    dis_out = netD(batch_fake_samples,batch_target_labels)
    g_loss = - torch.mean(torch.log(dis_out+1e-20))

    optimizerG.zero_grad()
    g_loss.backward()
    optimizerG.step()

    return netG

def gen_G(netG,batch_size,label):
    NFAKE = int(1e2)
    path = PROJECT_DIR+'/out/1DGAN/'

    fake_samples = np.zeros((NFAKE,output_shape[0],output_shape[1]),dtype=float)
    netG = netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            # print(tmp)
            z = torch.randn(batch_size,dim_latent_space,dtype=float).to(device)
            y = np.ones(batch_size) * label
            y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
            batch_fake_samples = netG(z.float(),y.float())
            batch_fake_samples = batch_fake_samples.view(-1,output_shape[0],output_shape[1])
            fake_samples[tmp:(tmp+batch_size)] = batch_fake_samples.cpu().detach().numpy()
            tmp += batch_size

    fake_samples = fake_samples[0:NFAKE]
    fake_labels = np.ones(NFAKE) * label

    mass_pair_str = "%04.0f,%1.2f"%(label,second_label)
    out_file_fstr = PROJECT_DIR+"/out/%s/gaus_%iepc_%s.%s" 

    if path is not None:
        for i in range(batch_size):
            fake_sample = fake_samples[i]
            # print(fake_sample)

            # out_file_jpg = out_file_fstr%("gen_jpg",mass_pair_str,"jpg")
            out_file_npy = out_file_fstr%("1DGAN/gen_npy",int(n_epochs),mass_pair_str+"_"+str(i),"npy")
            out_file_png = out_file_fstr%("1DGAN/gen_png",int(n_epochs),mass_pair_str+"_"+str(i),"png")

            np.save(out_file_npy,fake_sample,allow_pickle=False)
            plt.imsave(out_file_png,fake_sample.T,cmap="gray",vmin=0.,vmax=1.,format="png",origin="lower")

    return fake_samples,fake_labels



# start = timeit.default_timer()
# for epoch in range(n_epochs):
#     print("------------EPOCH: " + str(epoch) + " ------------")
#     netD = loss_D()
#     netD.float()
#     netG = loss_G()
#     netG.float()
# stop = timeit.default_timer()
# time_diff = stop-start
# print("training took: " + str(time_diff) + "s\t" + str(time_diff/60) + "m\t" + str(time_diff/3600) + "h")
# gen_G(netG,5,(0.6e3))