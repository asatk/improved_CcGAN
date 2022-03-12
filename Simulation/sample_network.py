import numpy as np
import os
import sys
from tqdm import tqdm
import torch
from utils import *
from models import *
from glob import glob
import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("filename", help=".pth file with GEN network")
parser.add_argument("label", help="")
parser.add_argument("num", type=int, help="number of samples")
parser.add_argument("-o", "--outfile", default='', help="filename for .npz file")
my_args = parser.parse_args()
sys.argv = [sys.argv[0]]
from Train_CcGAN import *
args = my_args

# network constants
radius = 1.
sigma_gaussian = 0.02
dim_gan = 2
n_features = 2

# load generator network
print("Load network...")
if not os.path.isfile(args.filename): raise SystemExit("Not a valid file")
Filename_GAN = args.filename
checkpoint = torch.load(Filename_GAN)
netG = cont_cond_generator(ngpu=NGPU, nz=dim_gan, out_dim=n_features, radius=radius).to(device)
netG.load_state_dict(checkpoint['netG_state_dict'])

# sample
print("Sample...")
def generate_data(n_samp_per_gaussian, angle_grid):
    return sampler_CircleGaussian(n_samp_per_gaussian, angle_grid, radius = radius, sigma = sigma_gaussian, dim = n_features)
nfake = args.num
batch_size = args.num
label = float(args.label)
gen_samples, _ = SampCcGAN_given_label(netG, label/(2*np.pi), path=None, NFAKE = nfake, batch_size = batch_size)
#gaus_samples, _, _ = generate_data(args.num, np.array([label]))
gaus_samples, _, _ = sampler_CircleGaussian(args.num, np.array([label]), radius = radius, sigma = sigma_gaussian, dim = n_features)

print("Writing out...")
out_file = 'val_L{}_N{}.npz'.format((args.label).replace('.','p'), args.num) if args.outfile=='' else args.outfile
np.savez(out_file, gen=gen_samples, gaus=gaus_samples)
