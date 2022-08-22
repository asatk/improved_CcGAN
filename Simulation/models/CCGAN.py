import numpy as np
import torch
import torch.nn as nn   

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

        # self.linear = nn.Linear(nz, self.inner_dim, bias=bias_g)

        self.main = nn.Sequential(
                nn.Linear(self.nz+2, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                # nn.ReLU(True),
                nn.LeakyReLU(inplace=True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                # nn.ReLU(True),
                nn.LeakyReLU(inplace=True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                # nn.ReLU(True),
                nn.LeakyReLU(inplace=True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                # nn.ReLU(True),
                nn.LeakyReLU(inplace=True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                # nn.ReLU(True),
                nn.LeakyReLU(inplace=True),

                # nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                # nn.BatchNorm1d(self.inner_dim),
                # nn.ReLU(True),

                nn.Linear(self.inner_dim, self.out_dim, bias=bias_g),
            )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = z.reshape(-1, self.nz)
        y = y.reshape(-1, 1)

        if self.geo == 'line':
            y = recover_labels_line_1d(y)
            z = torch.cat((z, torch.multiply(torch.ones((len(y), 1)), self.val), y), 1)
            # z = torch.cat((z, y, torch.multiply(torch.ones((len(y), 1)), self.val)), 1)
        elif self.geo == 'circle':
            y = y.reshape(-1, 1)*2*np.pi
            z = torch.cat((z, self.val*torch.sin(y), self.val*torch.cos(y)), 1)
        else:
            print("Only 'line' and 'circle' geometries are implemented for this network's forward pass")

        # #embed labels - NLI
        # output = self.linear(z) + y.repeat(1, self.inner_dim)
        # # output = output.reshape(-1, self.inner_dim)

        if z.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, z, range(self.ngpu))
            # output = nn.parallel.data_parallel(self.main, output, range(self.ngpu))
        else:
            output = self.main(z)
            # output = self.main(output)
        return output

#########################################################
# discriminator
bias_d=False
class discriminator(nn.Module):
    def __init__(self, ngpu=1, input_dim=2, val=1, geo='line'):
        super(discriminator, self).__init__()
        self.ngpu = ngpu
        self.input_dim = input_dim
        self.val = val
        self.geo = geo

        self.inner_dim = 100
        self.main = nn.Sequential(
            nn.Linear(input_dim+2, self.inner_dim, bias=bias_d),
            # nn.Linear(input_dim, self.inner_dim, bias=bias_d),
            # nn.ReLU(True),
            nn.LeakyReLU(inplace=True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d),
            # nn.ReLU(True),
            nn.LeakyReLU(inplace=True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d),
            # nn.ReLU(True),
            nn.LeakyReLU(inplace=True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d),
            # nn.ReLU(True),
            nn.LeakyReLU(inplace=True),

            # nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d),
            # nn.ReLU(True),
            # nn.LeakyReLU(inplace=True),

            nn.Linear(self.inner_dim, 1, bias=bias_d),
            nn.Sigmoid()
            # nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d)
        )

        # self.linear1 = nn.Linear(self.inner_dim, 1, bias=bias_d)
        # nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        # self.linear1 = nn.utils.parametrizations.spectral_norm(self.linear1)

        # self.linear2 = nn.Linear(1, self.inner_dim, bias=bias_d)
        # nn.init.xavier_uniform_(self.linear2.weight.data, 1.)
        # self.linear2 = nn.utils.parametrizations.spectral_norm(self.linear2)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.input_dim)
        y = y.reshape(-1, 1)

        if self.geo == 'line':
            y = recover_labels_line_1d(y)
            x = torch.cat((x, torch.multiply(torch.ones((len(y), 1)), self.val), y), 1)
            # x = torch.cat((x, y, torch.multiply(torch.ones((len(y), 1)), self.val)), 1)
        elif self.geo == 'circle':
            y = y.reshape(-1, 1)*2*np.pi
            x = torch.cat((x, self.val*torch.sin(y), self.val*torch.cos(y)), 1)
        else:
            print("Only 'line' and 'circle' geometries are implemented for this network's forward pass")        

        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)

        # #embed labels - NLI
        # output = output.reshape(-1, self.inner_dim)
        # output_y = torch.sum(output * self.linear2(y), 1, keepdim=True)
        # output = self.sigmoid(self.linear1(output) + output_y)

        return output.reshape(-1, 1)

if __name__ == "__main__":
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
