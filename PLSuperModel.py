import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F



class SuperModel(pl.LightningModule):
    def __init__(self, params, task, in_dim=(8, 159)):
        super().__init__()
        self.params = params
        self.task = task
        self.in_dim = in_dim
        if "masking" in params:
            self.masking = params["masking"]
        else:
            self.masking = False

        if params["activation"] == "relu":
            self.activation = nn.ReLU
        elif params["activation"] == "elu":
            self.activation = nn.ELU
        elif params["activation"] == "tanh":
            self.activation = nn.Tanh

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        return optimizer

    def weighted_mse_loss(self, input, target):
        weight = (target - target.mean()).abs()
        weight = weight.clamp(min=0.5)
        return torch.sum(weight * (input - target) ** 2)


    def loss_reg(self, y, y_hat, weights=None):
        if weights is None:
            return F.mse_loss(y_hat.flatten(), y) + self.params["l1_loss"] * F.l1_loss(y_hat.flatten(), y)
        else:
            return (F.mse_loss(y_hat.flatten(), y, reduction="none") * weights).mean() + self.params[
                "l1_loss"] * F.l1_loss(
                y_hat.flatten(), y)

    def loss_class(self, y, y_hat):
        weight = torch.ones(len(y))
        weight[y == 1] = 1  # self.ones_weight
        weight[y == 0] = 1  # self.zeros_weight
        weight = weight.to(self.device)
        # if self.num_of_classes == 2 or self.num_of_classes == 1:
        return F.binary_cross_entropy_with_logits(y_hat.flatten(), y.flatten(), weight=weight) + self.params[
            "l1_loss"] * F.l1_loss(y_hat.flatten(), y)


    def training_step(self, train_batch, batch_idx):
        all_loss = 0
        for tb in train_batch.keys():
            if len(train_batch[tb]) == 3:
                x, y, d = train_batch[tb]
                y_hat = self.forward(x, d)
            elif len(train_batch[tb]) == 4:
                x, y, d, a = train_batch[tb]
                y_hat = self.forward(x, torch.stack([d, a]))

            elif len(train_batch[tb]) == 5:
                x, y, d, a, m = train_batch[tb]
                y_hat = self.forward(x, torch.concat([d.unsqueeze(0), a.unsqueeze(0), m.T], dim=0))

            if self.masking:
                mask = d.T[1:, ]
                mask *= 4  # 9
                mask += 1

            if self.task == "reg":
                loss = self.loss_reg(y.type(torch.float32), y_hat, weights=mask if self.masking else None).type(
                    torch.float32)
            elif self.task == "class":
                loss = self.loss_class(y.type(torch.float32), y_hat).type(torch.float32)
            if tb == "fabricated":
                loss *= 1 / 1
            else:
                self.log("Loss", loss)
            all_loss += loss
        return all_loss

    def validation_step(self, valid_batch, batch_idx):
        all_loss = 0
        if len(valid_batch) == 3:
            x, y, d = valid_batch
            y_hat = self.forward(x, d)

        elif len(valid_batch) == 4:
            x, y, d, a = valid_batch
            y_hat = self.forward(x, torch.stack([d, a]))

        elif len(valid_batch) == 5:
            x, y, d, a, m = valid_batch
            y_hat = self.forward(x, torch.concat([d.unsqueeze(0), a.unsqueeze(0), m.T], dim=0))

        if self.task == "reg":
            loss = self.loss_reg(y.type(torch.float32), y_hat).type(torch.float32)
        elif self.task == "class":
            loss = self.loss_class(y.type(torch.float32), y_hat).type(torch.float32)
        self.log("val_Loss", loss, prog_bar=True)
        all_loss += loss
        return all_loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        y_hat = []
        proba = []
        if len(batch) == 3:
            x, y, d = batch
            if self.task == "reg":
                y_hat.extend([i.item() for i in self.forward(x, d).detach().cpu().numpy()])
            elif self.task == "class":
                if self.num_of_classes == 2:
                    y_hat.extend([torch.sigmoid(i).round().item() for i in self.forward(x, d).detach().cpu().numpy()])
                    proba.extend([torch.sigmoid(i).item() for i in self.forward(x, d).detach().cpu().numpy()])
            # return self.forward(x, d).detach().cpu().numpy()#org
        elif len(batch) == 4:
            x, y, d, a = batch
            if self.task == "reg":
                y_hat.extend([i.item() for i in self.forward(x, d).detach().cpu().numpy()])
            elif self.task == "class":
                y_hat.extend([torch.sigmoid(i).round().item() for i in
                              self.forward(x, torch.stack([d, a]))])
                proba.extend(
                    [torch.sigmoid(i).item() for i in self.forward(x, torch.stack([d, a]))])

        elif len(batch) == 5:
            x, y, d, a, m = batch
            if self.task == "reg":
                y_hat.extend([i.item() for i in self.forward(x, torch.concat([d.unsqueeze(0), a.unsqueeze(0), m.T], dim=0)).detach().cpu().numpy()])
            elif self.task == "class":
                # if self.num_of_classes == 2:
                y_hat.extend([torch.sigmoid(i).round().item() for i in
                              self.forward(x, torch.stack([d, a]))])
                proba.extend(
                    [torch.sigmoid(i).item() for i in self.forward(x, torch.stack([d, a]))])


        self.proba = proba
        return y_hat


