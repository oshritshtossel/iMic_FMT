import pandas as pd
import numpy as np
import torch
from numpy.random.mtrand import multivariate_normal


from GenerateData.CCGAN.MICECCGAN import MICECCGAN


def eval_a_div(counts):
    """
    calculate shannon alpha diversity
    :param counts: relative or abundences of bacterias
    :return: a_div
    """
    freqs = counts / counts.sum()
    nonzero_freqs = freqs[freqs != 0]
    return -(nonzero_freqs * np.log(nonzero_freqs)).sum() / np.log(2)


def generate_with_ccgan(donors, num_to_gen=450):
    mice = pd.read_csv("../a_divs/alpha_diversity_baby_allergy_gdm_chimo_mice_shannon.csv", index_col=0)
    mice = mice["0"]
    mice = mice[mice != 0]
    mice = mice.sample(num_to_gen, replace=True)

    days = [np.random.choice([3, 7, 14, 21, 28, 35, 42, 49]) for i in range(450)]
    # days_t = torch.tensor(days).type(torch.float32)

    cols = donors.columns
    mice_t = torch.tensor(mice.to_numpy())

    gan = MICECCGAN(159)
    gan = gan.load_from_checkpoint("../GenerateData/CCGAN/ccgan_model_v1.ckpt")

    z = torch.randn((num_to_gen, 159))

    out = gan(z.type(torch.float32), mice_t.type(torch.float32)).detach().numpy()

    out = pd.DataFrame(data=out, columns=cols)

    return out, mice, days
