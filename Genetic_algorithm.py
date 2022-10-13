import random


import pandas as pd
import numpy as np
import torch
from drawnow import drawnow
from scipy.stats import rv_histogram
import matplotlib as mpl

from CNN_model.CNN2convlayer import CNN
from microbiome2matrix import otu22d, dendogram_ordering

import matplotlib.pyplot as plt
import os


def create_binary_vectors(df):
    """
    Create binary vectors from the mean log vectors
    :param df: mean log abundances df
    :return: binary df where all the zeros bacterias (with value -1 = log(0.1) turns to 0 and all other bacterias are 1.
    """
    binary = df.replace([-1], 0.)
    binary[binary != 0] = 1.
    return binary


def sample_from_real_dist(org_df, binary_df, samples=100):
    new_sample = pd.DataFrame(columns=org_df.columns, index=range(samples), dtype=float)
    for bact in org_df:
        no_minus_1_bact = org_df[bact][org_df[bact] != -1.]
        if len(no_minus_1_bact) == 0:
            new_sample[bact] = -1.
            continue
        hist = np.histogram(no_minus_1_bact, 20)
        hist_dist = rv_histogram(hist)
        for i in range(samples):
            r = random.random()
            ppf = hist_dist.ppf(r)
            new_sample[bact][i] = ppf

    for i in range(samples):
        eql = binary_df.sample(1).iloc[0]
        new_sample.iloc[i][eql == 0] = -1.

    return new_sample


def load_model(shape):
    D = {
        "l1_loss": 0.5634906869223949,
        "weight_decay": 0.0005552445335025638,
        "lr": 0.001,
        "batch_size": 128,
        "activation": "relu",
        "dropout": 0.1,
        "kernel_size_a": 4,
        "kernel_size_b": 12,
        "stride": 1,
        "padding": 2,
        "padding_2": 0,
        "kernel_size_a_2": 2,
        "kernel_size_b_2": 5,
        "stride_2": 1,
        "channels": 7,
        "channels_2": 4,
        "linear_dim_divider_1": 6,
        "linear_dim_divider_2": 11
    }
    model = CNN(D, in_dim=shape)
    model = model.load_from_checkpoint("CNN_model/cnn2_logmean_weights.ckpt", in_dim=shape, params=D)
    return model


def mutation(sample: pd.Series):
    try:
        zero = sample[sample == -1].sample(1)
        non_zero = sample[sample != -1].sample(1)

        sample[zero.index[0]] = non_zero.values[0]
        sample[non_zero.index[0]] = zero.values[0]
    except ValueError:
        pass
    return sample


def recombine(sample: pd.Series, comb_with: pd.Series):
    sample = sample * 0.5 + comb_with * 0.5
    return sample


def sample_from_gen_dist(generated_donors, samples):
    binary = create_binary_vectors(generated_donors)
    new_sampels = sample_from_real_dist(generated_donors, binary, samples=samples)
    for i, sample in enumerate(new_sampels.iloc):
        r = random.random()
        if r < 0.33:
            new_sampels.at[i] = mutation(sample)
        r = random.random()
        if r < 0.33:
            new_sampels.at[i] = recombine(sample, new_sampels.sample(1).iloc[0])
    return new_sampels


def main(gamma=0., mode="max"):
    donors = pd.read_csv('tax_encoding/tax_7_log_mean.csv', index_col=0)
    binary = create_binary_vectors(donors)
    model = load_model((8, 254))
    num_of_generated = 200

    new_sampels = sample_from_real_dist(donors, binary, samples=num_of_generated)
    max_in_gen = []
    mean_of_best_in_gen = []
    max_bact_in_gen = []
    min_bact_in_gen = []
    mean_bact_in_gen = []
    plt.ion()  # enable interactivity
    fig_a = plt.figure("alpha_div")  # make a figure
    fig_b = plt.figure("num_of_non_zero")

    mode = mode

    for i in range(25):

        otus2d, names = otu22d(new_sampels, with_names=True, save=False)
        otus2d, dendogramed_df = dendogram_ordering(otus2d, new_sampels, names, save=False)
        otus_tensor = torch.tensor(otus2d)

        days = torch.ones(otus2d.shape[0]) * 7
        pred = model(otus_tensor, days).detach().flatten().numpy()
        pd.DataFrame(data=pred).to_csv("final_fig/Fig_3/orders/preds_Shannon.csv")

        non_zero_pen = (new_sampels != -1.).sum(axis=1)

        best_30_p = int((num_of_generated / 10) * 3)
        if mode == "max":
            loss = pred - non_zero_pen * gamma
            best_30_p_places = np.argpartition(loss, -best_30_p)[-best_30_p:]
        elif mode == "min":
            loss = pred + non_zero_pen * gamma
            best_30_p_places = np.argpartition(loss, best_30_p)[:best_30_p]

        selected = new_sampels.iloc[best_30_p_places]
        new_sampels = sample_from_gen_dist(selected, num_of_generated)

        max_bact_in_gen.append(non_zero_pen[best_30_p_places].max())
        min_bact_in_gen.append(non_zero_pen[best_30_p_places].min())
        mean_bact_in_gen.append(non_zero_pen[best_30_p_places].mean())
        if mode == "max":
            max_in_gen.append(pred[best_30_p_places].max())
        elif mode == "min":
            max_in_gen.append(pred[best_30_p_places].min())

        mean_of_best_in_gen.append(pred[best_30_p_places].mean())

        def make_fig():
            plt.figure(figsize=(4,4))
            plt.plot(max_in_gen, c="blue", label=mode)
            plt.plot(mean_of_best_in_gen, c="red", label="mean")
            plt.xlabel("Generation",fontdict={"fontsize": 15})
            plt.ylabel("Shannon index",fontdict={"fontsize": 15})
            plt.legend()

        plt.figure("alpha_div")
        drawnow(make_fig)

        def make_fig():
            plt.figure(figsize=(4, 4))
            plt.plot(max_bact_in_gen, c="blue", label="max")
            plt.plot(mean_bact_in_gen, c="red", label="mean")
            plt.plot(min_bact_in_gen, c="blue", label="min")
            plt.xlabel("Generation",fontdict={"fontsize": 15})
            plt.ylabel("Number of non-zero taxa",fontdict={"fontsize": 15})
            plt.legend()

        plt.figure("num_of_non_zero")
        drawnow(make_fig)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    selected = (10. ** selected) - 0.1
    selected[selected < 0.0] = 0.
    selected.to_csv(f"final_fig/Fig_3/data/{mode}/chosen_{gamma}_1__.csv")

    plt.figure("alpha_div")
    plt.savefig(f"final_fig/Fig_3/{mode}/final_genetic_algo_{mode}_{gamma}.png")

    plt.figure("num_of_non_zero")
    plt.savefig(f"final_fig/Fig_3/{mode}/final_num_non_zero_bact_{mode}_{gamma}.png")


if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')
    for g in [0.]:#[0.015, 0.02, 0.03, 0.035, 0.04, 0.045]:
        for m in ['max']:
            main(g, m)
