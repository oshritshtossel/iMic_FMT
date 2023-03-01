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


class CNN(SuperModel):
    def __init__(self, params, in_dim, task="reg"):
        super().__init__(params, in_dim, task)
        self.task = task
        self.cnn = nn.Sequential(
            nn.Conv2d(1, params["channels"], kernel_size=(params["kernel_size_a"], params["kernel_size_b"]),
                      stride=params["stride"], padding=params["padding"]),
            self.activation(),

            nn.Conv2d(params["channels"], params["channels_2"],
                      kernel_size=(params["kernel_size_a_2"], params["kernel_size_b_2"]),
                      stride=params["stride_2"], padding=params["padding_2"]),
            self.activation(),
        )

        cos1 = conv_output_shape(in_dim, (params["kernel_size_a"], params["kernel_size_b"]), stride=params["stride"],
                                 pad=params["padding"])
        cos = conv_output_shape(cos1, (params["kernel_size_a_2"], params["kernel_size_b_2"]), stride=params["stride_2"],
                                pad=params["padding_2"])

        # 2 if alpha_donors isnot None
        # 5 if meta_is not none

        conv_out_dim = int(cos[0] * cos[1] * params["channels_2"]) + 2 + (1 if "masking" in params and params[
            "masking"] == True else 0)

        self.lin = nn.Sequential(
            nn.Linear(conv_out_dim, conv_out_dim // params["linear_dim_divider_1"]),
            self.activation(),
            nn.Dropout(params["dropout"]),
            nn.Linear(conv_out_dim // params["linear_dim_divider_1"], conv_out_dim // params["linear_dim_divider_2"]),
            self.activation(),
            nn.Dropout(params["dropout"]),
            nn.Linear(conv_out_dim // params["linear_dim_divider_2"], 1)
            # nn.Linear(conv_out_dim // params["linear_dim_divider_2"], 210)
        )

    def forward(self, x, d):
        x = x.type(torch.float32)
        x = torch.unsqueeze(x, 1)
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, d.unsqueeze(1) if len(d.shape) == 1 else d.T], dim=1).type(torch.float32)
        # x = torch.cat([x, d.unsqueeze(1) if len(d.shape) == 1 else d.T], dim=1).type(torch.float32)
        x = self.lin(x).type(torch.float32)

        # if self.params["rank"] == True:
        #     x = torch.softmax(x, 0)
        return x
