
import pytorch_lightning as pl
import numpy as np
import torch


from GenerateData.CCGAN.Gen_and_Dis_CCGAN import Generator, Discriminator


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


def eval_a_div(counts):
    tensor_mode = False
    if isinstance(counts, torch.Tensor):
        tensor_mode = True
        counts = counts.squeeze()
        counts = counts.detach().numpy()
    if len(counts.shape) == 1:
        freqs = counts / counts.sum()
        nonzero_freqs = freqs[freqs != 0]
        return torch.tensor([-(nonzero_freqs * np.log(nonzero_freqs)).sum() / np.log(2)])

    all_ret = torch.tensor([])
    for c in counts:
        freqs = c / c.sum()
        nonzero_freqs = freqs[freqs != 0]
        ret = -(nonzero_freqs * np.log(nonzero_freqs)).sum() / np.log(2)
        all_ret = torch.cat([all_ret, torch.tensor(ret).unsqueeze(0)])
    return all_ret


class MICECCGAN(pl.LightningModule):
    def __init__(
            self,
            im_size,
            threshold_type="hard",
            learning_rate: float = 0.0002,
            **kwargs
    ):
        super().__init__()

        # makes self.hparams under the hood and saves to ckpt
        self.save_hyperparameters()
        self.threshold_type = threshold_type
        # self.img_dim = input_width  # (input_channels, input_height, input_width)

        # networks
        self.generator = self.init_generator(im_size)
        self.discriminator = self.init_discriminator(im_size)

    def init_generator(self, im_size):
        generator = Generator(im_size)
        return generator

    def init_discriminator(self, im_size):
        discriminator = Discriminator(im_size)
        return discriminator

    def forward(self, z, y):
        """
        Generates an image given input noise z
        Example::
            z = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(z)
        """
        z = z.squeeze()
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        return self.generator(z, y)

    def on_train_start(self) -> None:
        adivs = []
        for D in self.train_dataloader.dataloader:
            mice_adiv = D[1]
            adivs.extend(mice_adiv)

        adivs = torch.tensor(adivs)
        self.a_div_mean = adivs.mean()
        self.a_div_var = adivs.std() ** 2

    def generator_loss(self, donor, mice_adiv, fab):
        batch_size = mice_adiv.shape[0]

        y_sorted = np.sort(mice_adiv)
        # todo: add random sampling to y_sorted

        kernal_sigma = ((4 * (mice_adiv.std() ** 5)) / (3 * batch_size)) ** (1 / 5)
        b_epsilon = np.random.normal(0, kernal_sigma, batch_size)
        batch_target_labels = y_sorted.squeeze() + b_epsilon
        batch_target_labels[batch_target_labels < 0] = batch_target_labels[batch_target_labels < 0] + 1
        batch_target_labels[batch_target_labels > 1] = batch_target_labels[batch_target_labels > 1] - 1
        batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float)
        if fab is None:
            z = torch.randn(donor.shape)
            batch_fake_samples = self(z, batch_target_labels)
        else:
            batch_fake_samples = self(fab, batch_target_labels)

        dis_out = self.discriminator(batch_fake_samples, batch_target_labels)
        g_loss = - torch.mean(torch.log(dis_out + 1e-20))

        return g_loss

    def discriminator_loss(self, donors, adiv_mice, fab):
        batch_size = adiv_mice.shape[0]

        adiv_mice_no_neg = adiv_mice[adiv_mice != -1]
        batch_size_no_neg = len(adiv_mice_no_neg)
        donors_no_neg = donors[(adiv_mice != -1).squeeze()]

        y_sorted = np.sort(adiv_mice_no_neg)
        if batch_size_no_neg <= 1:
            return torch.tensor(0.0, requires_grad=True)
        # todo: add random sampling to y_sorted

        # todo: normalize labels
        kernal_sigma = ((4 * (adiv_mice_no_neg.std() ** 5)) / (3 * batch_size_no_neg)) ** (1 / 5)
        b_epsilon = np.random.normal(0, kernal_sigma, 1)
        b_target = y_sorted.squeeze() + b_epsilon
        kappa = max([y_sorted[i + 1] - y_sorted[i] for i in range(batch_size_no_neg - 1)])
        b_target[b_target < 0] = b_target[b_target < 0] + 1
        b_target[b_target > 1] = b_target[b_target > 1] - 1

        batch_real_indx = torch.zeros(batch_size_no_neg, dtype=torch.int64)
        batch_fake_labels = torch.zeros(batch_size_no_neg)

        for j in range(batch_size_no_neg):
            if self.threshold_type == "hard":
                indx_real_in_vicinity = np.where(np.abs(adiv_mice_no_neg - b_target[j]).detach().numpy() <= kappa)[0]
            else:
                # todo: what is this 1? "nonzero_soft_weight_threshold"?
                indx_real_in_vicinity = np.where(
                    (adiv_mice_no_neg - b_target[j]) ** 2 <= -np.log(1) / kappa)[0]
            while len(indx_real_in_vicinity) < 1:
                b_epsilon_j = np.random.normal(0, kernal_sigma, 1)
                b_target[j] = y_sorted[j] + b_epsilon_j
                if b_target[j] < 0:
                    b_target[j] = b_target[j] + 1
                if b_target[j] > 1:
                    b_target[j] = b_target[j] - 1
                if self.threshold_type == "hard":
                    indx_real_in_vicinity = np.where(np.abs(adiv_mice_no_neg - b_target[j]).detach().numpy() <= kappa)[
                        0]
                else:
                    # todo: what is this 1? "nonzero_soft_weight_threshold"?
                    indx_real_in_vicinity = np.where(
                        (adiv_mice_no_neg - b_target[j]) ** 2 <= -np.log(1) / kappa)[0]
            assert len(indx_real_in_vicinity) >= 1

            batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]

            if self.threshold_type == "hard":
                lb = b_target[j] - kappa
                ub = b_target[j] + kappa
            else:
                # todo: what is this 1? "nonzero_soft_weight_threshold"?
                lb = b_target[j] - np.sqrt(-np.log(1) / kappa)
                ub = b_target[j] + np.sqrt(-np.log(1) / kappa)
            lb = max(0.0, lb)
            ub = min(1.0, ub)
            assert lb <= ub
            assert lb >= 0 and ub >= 0
            assert lb <= 1 and ub <= 1
            batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]

        batch_real_samples = donors_no_neg[batch_real_indx]
        batch_real_labels = adiv_mice_no_neg[batch_real_indx]
        # batch_real_samples = torch.from_numpy(batch_real_samples).type(torch.float)
        # batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float)
        # batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float)

        if fab is None:
            z = torch.randn(donors_no_neg.shape)
            batch_fake_samples = self(z, batch_fake_labels)
        else:
            batch_fake_samples = self(fab, batch_fake_labels)

        batch_target_labels = torch.from_numpy(b_target).type(torch.float)

        if self.threshold_type == "soft":
            real_weights = torch.exp(-kappa * (batch_real_labels - batch_target_labels) ** 2)
            fake_weights = torch.exp(-kappa * (batch_fake_labels - batch_target_labels) ** 2)
        else:
            real_weights = torch.ones(batch_size_no_neg, dtype=torch.float)
            fake_weights = torch.ones(batch_size_no_neg, dtype=torch.float)

        real_dis_out = self.discriminator(batch_real_samples, batch_target_labels)
        fake_dis_out = self.discriminator(batch_fake_samples.detach(), batch_target_labels)
        d_loss = - torch.mean(real_weights.view(-1) * torch.log(real_dis_out.view(-1) + 1e-20)) - torch.mean(
            fake_weights.view(-1) * torch.log(1 - fake_dis_out.view(-1) + 1e-20))
        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        if len(batch) == 3:
            donor, adiv_mice, fab = batch
        else:
            donor, adiv_mice = batch
            fab = None
        # train generator

        if optimizer_idx == 0:
            loss = self.generator_step(donor, adiv_mice, fab)

        # train discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(donor, adiv_mice, fab)

        return loss

    def generator_step(self, donor, adiv_mice, fab):
        g_loss = self.generator_loss(donor, adiv_mice, fab)

        # log to prog bar on each step AND for the full epoch
        # use the generator loss for checkpointing
        # result = pl.TrainResult(minimize=g_loss, checkpoint_on=g_loss)
        self.log('g_loss', g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def discriminator_step(self, donors, adiv_mice, fab):
        # Measure discriminator's ability to classify real from generated samples
        d_loss = self.discriminator_loss(donors, adiv_mice, fab)

        # log to prog bar on each step AND for the full epoch
        # result = pl.TrainResult(minimize=d_loss)
        self.log('d_loss', d_loss, on_epoch=True, prog_bar=True)
        return d_loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
                                 )
        return [opt_g, opt_d], []
