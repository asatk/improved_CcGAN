'''
Definitions of utility functions for analyzing CCGAN.

Author: Anthony Atkinson
'''

from models.CCGAN import generator
import numpy as np
from PIL import Image
import torch

def sample_gen_for_label(gen: generator, n_samples: int, label: float, path: str=None, batch_size: int=500, n_dim: int=2) -> tuple[np.ndarray, np.ndarray]:
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