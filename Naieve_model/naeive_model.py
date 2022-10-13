from torch import nn
import torch


from PLSuperModel import SuperModel


class Naeive(SuperModel):
    def __init__(self, params, in_dim=160):
        super().__init__(params, in_dim)
        self.params = params

        self.lin = nn.Sequential(
            nn.Linear(in_dim, params["linear_dim_1"]),
            self.activation(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["linear_dim_1"], params["linear_dim_2"]),
            self.activation(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["linear_dim_2"], 1)
        )

    def forward(self, x, d):
        x = x.type(torch.float32)
        x = torch.cat([x, d.unsqueeze(1)], dim=1).type(torch.float32)
        x = self.lin(x).type(torch.float32)
        return x
