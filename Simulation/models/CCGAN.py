import torch
import torch.nn as nn
import numpy as np

from train_utils import recover_labels_line_1d

#########################################################
# generator
bias_g = False
class generator(nn.Module):
    def __init__(self, ngpu=1, nz=2, out_dim=2, val=0.5, geo='line'):
        super(generator, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.out_dim = out_dim
        self.val = val
        self.geo = geo

        self.inner_dim = 100

        self.linear = nn.Sequential(
                nn.Linear(nz+2, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.out_dim, bias=bias_g),
            )

    def forward(self, z, labels):
        z = z.reshape(-1, self.nz)
        labels = labels.reshape(-1, 1)

        if self.geo == 'line':
            labels = recover_labels_line_1d(labels)
            z = torch.cat((z, torch.multiply(torch.ones((len(labels), 1)), self.val), labels), 1)
            # z = torch.cat((z, labels, torch.multiply(torch.ones((len(labels), 1)), self.val)), 1)
        elif self.geo == 'circle':
            labels = labels.reshape(-1, 1)*2*np.pi
            z = torch.cat((z, self.val*torch.sin(labels), self.val*torch.cos(labels)), 1)
        else:
            print("Only 'line' and 'circle' geometries are implemented for this network's forward pass")

        if z.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.linear, z, range(self.ngpu))
        else:
            output = self.linear(z)
        return output

#########################################################
# discriminator
bias_d=False
class discriminator(nn.Module):
    def __init__(self, ngpu=1, input_dim = 2, val=1, geo='line'):
        super(discriminator, self).__init__()
        self.ngpu = ngpu
        self.input_dim = input_dim
        self.val = val
        self.geo = geo

        self.inner_dim = 100
        self.main = nn.Sequential(
            nn.Linear(input_dim+2, self.inner_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, 1, bias=bias_d),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.reshape(-1, self.input_dim)
        labels = labels.reshape(-1, 1)

        if self.geo == 'line':
            labels = recover_labels_line_1d(labels)
            x = torch.cat((x, torch.multiply(torch.ones((len(labels), 1)), self.val), labels), 1)
            # x = torch.cat((x, labels, torch.multiply(torch.ones((len(labels), 1)), self.val)), 1)
        elif self.geo == 'circle':
            labels = labels.reshape(-1, 1)*2*np.pi
            x = torch.cat((x, self.val*torch.sin(labels), self.val*torch.cos(labels)), 1)
        else:
            print("Only 'line' and 'circle' geometries are implemented for this network's forward pass")

        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output.reshape(-1, 1)

if __name__=="__main__":
    import numpy as np
    #test
    ngpu=1

    netG = generator(ngpu=ngpu, nz=2, out_dim=2).cuda()
    netD = discriminator(ngpu=ngpu, input_dim = 2).cuda()

    z = torch.randn(32, 2).cuda()
    y = np.random.randint(100, 300, 32)
    y = torch.from_numpy(y).type(torch.float).reshape(-1,1).cuda()
    x = netG(z,y)
    o = netD(x,y)
    print(y.size())
    print(x.size())
    print(o.size())
