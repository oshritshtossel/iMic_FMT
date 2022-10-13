import nni
import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F
from nni.nas.pytorch import mutables

from PLSuperModel import SuperModel

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class CNN_1l(SuperModel):
    def __init__(self, params, in_dim):
        super().__init__(params, in_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, params["channels"], kernel_size=(params["kernel_size_a"], params["kernel_size_b"]),
                      stride=params["stride"]),
            self.activation()
        )

        cos = conv_output_shape(in_dim, (params["kernel_size_a"], params["kernel_size_b"]), stride=params["stride"])
        conv_out_dim = int(cos[0] * cos[1] * params["channels"]) + 1

        self.lin = nn.Sequential(
            nn.Linear(conv_out_dim, conv_out_dim // params["linear_dim_divider_1"]),
            self.activation(),
            # nn.Dropout(params["dropout"]),
            nn.Linear(conv_out_dim // params["linear_dim_divider_1"], conv_out_dim // params["linear_dim_divider_2"]),
            self.activation(),
            nn.Dropout(params["dropout"]),
            nn.Linear(conv_out_dim // params["linear_dim_divider_2"], 1)
        )

    def forward(self, x, d):
        x = x.type(torch.float32)
        x = torch.unsqueeze(x, 1)
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, d.unsqueeze(1)], dim=1).type(torch.float32)
        return self.lin(x).type(torch.float32)
