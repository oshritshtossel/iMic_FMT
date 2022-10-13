import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torch

from torch import nn


class Generator(nn.Module):
    '''
    Generator class in a CGAN. Accepts a noise tensor (latent dim 100)
    and a label tensor as input as outputs another tensor of size 784.
    Objective is to generate an output tensor that is indistinguishable
    from the real MNIST digits.
    '''

    def __init__(self,im_size, radius=1):
        super().__init__()
        self.nz = im_size
        self.out_dim = im_size
        self.radius = radius
        self.inner_dim = 100
        bias_g = False

        self.linear = nn.Sequential(
            nn.Linear(im_size + 2, self.inner_dim, bias=bias_g),
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

    def forward(self, input, labels):
        # input = input.view(-1, self.nz)
        labels = labels.view(-1, 1) * 2 * np.pi
        input = torch.cat((input, self.radius * torch.sin(labels), self.radius * torch.cos(labels)), 1)

        output = self.linear(input)
        return output


class Discriminator(nn.Module):
    '''
    Discriminator class in a CGAN. Accepts a tensor of size 784 and
    a label tensor as input and outputs a tensor of size 1,
    with the predicted class probabilities (generated or real data)
    '''

    def __init__(self, im_size, radius=1):
        super().__init__()
        self.input_dim = im_size
        self.radius = radius
        self.inner_dim = 100
        bias_d = False

        self.main = nn.Sequential(
            nn.Linear(self.input_dim + 2, self.inner_dim, bias=bias_d),
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

    def forward(self, input, labels):
        # input = input.view(-1, self.input_dim)

        labels = labels.view(-1, 1) * 2 * np.pi
        input = torch.cat((input, self.radius * torch.sin(labels), self.radius * torch.cos(labels)), 1)

        output = self.main(input)
        return output.view(-1, 1)
