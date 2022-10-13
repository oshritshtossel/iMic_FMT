import itertools
import random
from collections import defaultdict
import sys
from operator import not_
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
sys.path.insert(0, "..")

from nni_data_loader import load_nni_data
from nni.algorithms.nas.pytorch.classic_nas import get_and_apply_next_architecture
import os
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from GenerateData.CCGAN.generate_ccgan import generate_with_ccgan
from microbiome2matrix import otu22d, dendogram_ordering
from DataAugment.random_augment import randomAugment
from CNN_model.CNN1convlayer import CNN_1l
from Naieve_model.naeive_model import Naeive
import nni
import pandas as pd
import numpy as np
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import spearmanr, pearsonr
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from CNN_model.CNN2convlayer import CNN
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit

# from utils import amax2

from torch.utils import data





def PairwiseDataSet(dataset):
    d = list(itertools.product(dataset, dataset))
    d = ([(*i[0], *i[1]) for i in d])
    D = []
    for i in range(len(d[0])):
        D.append(torch.stack([t[i] for t in d]))
    return data.TensorDataset(*D)


def POC(train_dataset, valid_dataset, test_dataset, y_train, y_valid, y_test, fab_dataset=None,
        model: pl.LightningModule = None, parms: dict = None, test=True, input_dim=None, pairwise=False,
        additional_train_dataset=None):
    """
    a. try model according to its parms on data
    b. calculate Spearman correlation
    c. calculate R2

    :param: train_dataset: x_train
    :param: valid_dataset: x_valid
    :param: test_dataset: x_test
    :param: y_train:
    :param: y_valid:
    :param: y_test:
    :model: learning model we want to use, must be a pytorch lightening model
    :param: parms: hyper parameters dictionary
    :param: test: bool which says whether to do test prediction and calculate its corr and R2
    :return: r2_tr, r2_val, c_tr, c_val


    """
    num_workers = 0
    if torch.cuda.is_available():
        num_workers = 4
    # load data according to batches:
    if pairwise:
        train_dataset = PairwiseDataSet(train_dataset)
        test_dataset = PairwiseDataSet(test_dataset)
        valid_dataset = PairwiseDataSet(valid_dataset)
    trainloader = data.DataLoader(train_dataset, batch_size=parms["batch_size"], num_workers=num_workers, shuffle=True)
    testloader = data.DataLoader(test_dataset, batch_size=parms["batch_size"], num_workers=num_workers)
    if valid_dataset != None:
        validloader = data.DataLoader(valid_dataset, batch_size=parms["batch_size"], num_workers=num_workers)
        mc = ModelCheckpoint("lightning_logs", monitor='val_Loss')  # "lightning_logs", monitor='val_Loss'
    else:
        mc = None

    trainDataloaders = {"main": trainloader}
    if fab_dataset is not None:
        fabLoader = data.DataLoader(fab_dataset, batch_size=parms["batch_size"], num_workers=num_workers)
        trainDataloaders["fabricated"] = fabLoader
    if additional_train_dataset is not None:
        addLoader = data.DataLoader(additional_train_dataset, batch_size=parms["batch_size"], num_workers=num_workers)
        trainDataloaders["additional"] = addLoader

    # get_and_apply_next_architecture(model)

    # early stopping when there is no change in val loss for 20 epochs, where no change is defined according
    # to min_delta
    callbacks = []
    if valid_dataset is not None:
        callbacks.append(EarlyStopping(monitor='val_Loss', patience=20, min_delta=0.0001, mode="min"))

    if mc is not None:
        callbacks.append(mc)

    if torch.cuda.is_available():
        tt = pl.Trainer(precision=32, callbacks=callbacks, gpus=1, reload_dataloaders_every_epoch=False)
    else:
        # tt = pl.Trainer(precision=32, logger=TensorBoardLogger(r'D:\lightning_logs'), max_epochs=250,
        #                 callbacks=callbacks,checkpoint_callback=mc)
        tt = pl.Trainer(precision=32, logger=TensorBoardLogger("lightning_logs"), max_epochs=250,
                        callbacks=callbacks, checkpoint_callback=mc)

    try:
        model._estimator_type
        del parms["batch_size"]
        model = model(**parms)
        model.fit(train_dataset.tensors[0].numpy(), train_dataset.tensors[1].numpy())

        # model = model.load_from_checkpoint(mc.best_model_path)

        pred_train = model.predict(train_dataset.tensors[0].numpy())
        pred_valid = model.predict(valid_dataset.tensors[0].numpy())
        if test:
            pred_test = model.predict(test_dataset.tensors[0].numpy())

        plt.hist(train_dataset.tensors[1].numpy().squeeze(), bins=30, color="orange", label="Real")
        to_save = pd.DataFrame(columns=["real", "pred"])
        to_save["real"] = test_dataset.tensors[1].numpy().squeeze()
        to_save["pred"] = pred_test.squeeze()
        to_save.to_csv(f"../final_fig/Fig_2/species/rf_test_{specie_name}_{e}.csv")


    except:
        model = model(parms, input_dim)
        tt.fit(model, trainDataloaders, validloader if valid_dataset is not None else None)

        trainloader_no_shuffle = data.DataLoader(train_dataset, batch_size=parms["batch_size"], num_workers=num_workers,
                                                 shuffle=False)

        pred_train = []
        pred_tr = tt.predict(model, trainloader_no_shuffle)
        if not pairwise:
            [pred_train.extend(i) for i in pred_tr]
            pred_train = np.array(pred_train)

        if valid_dataset is not None:
            pred_valid = []
            pred_v = tt.predict(model, validloader)
            if not pairwise:
                [pred_valid.extend(i) for i in pred_v]
                pred_valid = np.array(pred_valid)

        if test:
            pred_test = []
            pred_te = tt.predict(model, testloader)
            if not pairwise:
                [pred_test.extend(i) for i in pred_te]
                pred_test = np.array(pred_test)

        ##################################################3
        # iMic
        fig, ax = plt.subplots()
        min1 = train_dataset.tensors[1].numpy().squeeze().min()
        max1 = train_dataset.tensors[1].numpy().squeeze().max()
        min2 = pred_train.squeeze().min()
        max2 = pred_train.squeeze().max()

        lims = [
            np.min([min1, min2]),  # min of both axes
            np.max([max1, max2]),  # max of both axes
        ]
        corr = spearmanr(train_dataset.tensors[1].numpy().squeeze(), pred_train.squeeze())[0]
        ax.scatter(pred_train.squeeze(), train_dataset.tensors[1].numpy().squeeze(), alpha=0.7,
                   label=f"SCC:{round(corr, 3)}", color="magenta", )
        ax.plot(lims, lims, 'k-', zorder=0)
        plt.xlabel("Predicted mice shannon", fontdict={"fontsize": 20})
        plt.ylabel("Real mice shannon", fontdict={"fontsize": 20})
        # plt.title(name_)

        plt.legend(prop={'size': 15})
        plt.tight_layout()
        # plt.savefig(f"../final_fig/Fig_2/species/train_{specie_name}_{e}.png")
        print(spearmanr(train_dataset.tensors[1].numpy().squeeze(), pred_train.squeeze()))
        # plt.show()
        plt.clf()

        fig, ax = plt.subplots()
        min1 = test_dataset.tensors[1].numpy().squeeze().min()
        max1 = test_dataset.tensors[1].numpy().squeeze().max()
        min2 = pred_test.squeeze().min()
        max2 = pred_test.squeeze().max()

        lims = [
            np.min([min1, min2]),  # min of both axes
            np.max([max1, max2]),  # max of both axes
        ]
        corr = spearmanr(test_dataset.tensors[1].numpy().squeeze(), pred_test.squeeze())[0]
        ax.scatter(pred_test.squeeze(), test_dataset.tensors[1].numpy().squeeze(), alpha=0.7, color="magenta",
                   label=f"SCC:{round(corr, 3)}")
        to_save = pd.DataFrame(columns=["real", "pred"])
        to_save["real"] = test_dataset.tensors[1].numpy().squeeze()
        to_save["pred"] = pred_test.squeeze()
        print(pearsonr(to_save["real"], to_save["pred"]))
        to_save.to_csv(f"../final_fig/Fig_2/order_meta/{name_}_test_{e}.csv")
        ax.plot(lims, lims, 'k-', zorder=0)
        plt.xlabel("Predicted mice shannon", fontdict={"fontsize": 20})
        plt.ylabel("Real mice shannon", fontdict={"fontsize": 20})
        # plt.title(name_)

        plt.legend(prop={'size': 15})
        plt.tight_layout()
        # plt.savefig(f"../final_fig/Fig_2/a_div_test_{e}_.png")
        print(spearmanr(train_dataset.tensors[1].numpy().squeeze(), pred_train.squeeze()))
        # plt.show()



    if pairwise:
        acc = None
        if test:
            pred_te = np.hstack(pred_te)
            acc = (pred_te[0] == pred_te[1]).sum() / pred_te.shape[1]
        pred_tr = np.hstack(pred_tr)
        acc_tr = (pred_tr[0] == pred_tr[1]).sum() / pred_tr.shape[1]
        if valid_dataset is not None:
            pred_v = np.hstack(pred_v)
            acc_val = (pred_v[0] == pred_v[1]).sum() / pred_v.shape[1]
        return acc_tr, acc_val if valid_dataset is not None else None, 0, 0 if valid_dataset is not None else None, 0, 0 if valid_dataset is not None else None, acc, 0

    r2 = None
    c = None
    if test:
        r2 = r2_score(y_test, pred_test)
        c = spearmanr(y_test, pred_test)[0]
    r2_tr = r2_score(y_train, pred_train)
    c_tr = spearmanr(y_train, pred_train)[0]
    if valid_dataset is not None:
        r2_val = r2_score(y_valid, pred_valid)
        c_val = spearmanr(y_valid, pred_valid)[0]

    r2_tr_round = r2_score(np.round(y_train), np.round(pred_train))
    if valid_dataset is not None:
        r2_val_round = r2_score(np.round(y_valid), np.round(pred_valid))

    if valid_dataset is None:
        return r2_tr, c_tr, pred_test

    return r2_tr, r2_val if valid_dataset is not None else None, c_tr, c_val if valid_dataset is not None else None, r2_tr_round, r2_val_round if valid_dataset is not None else None, r2, c




def projection(df: pd.DataFrame):
    """
    draws the samples in a 3 dim graph where the samples of each exp are in a different color
    :param df: 3 dim pca from mip mlp
    :return: None
    """
    expers_dim1 = defaultdict(list)
    expers_dim2 = defaultdict(list)
    expers_dim3 = defaultdict(list)
    for i in df.iloc:
        expers_dim1[i.name[:i.name.find('_')]].append(i[0])
        expers_dim2[i.name[:i.name.find('_')]].append(i[1])
        try:
            expers_dim3[i.name[:i.name.find('_')]].append(i[2])
        except:
            pass
    colors = ['r', 'b', 'g', 'purple', 'orange', 'black']
    if len(expers_dim3) == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, k in enumerate(expers_dim1):
            ax.scatter(expers_dim1[k], expers_dim2[k], c=colors[i], label=k)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, k in enumerate(expers_dim1):
            ax.scatter(expers_dim1[k], expers_dim2[k], expers_dim3[k], c=
            colors[i], label=k)
    plt.legend()
    plt.show()


def load_data_2d_train_test(donors, mapping, a_div, path, augmented_data=None, donors_grouped=True, test_size=0.15,
                            alpha_donors=None, meta_=None):  # 0.2519809825673534
    """
    a. load 2 dimensional microbiome data days after transplant data and tag data
    according to mapping file
    b. split train and test, while the test is 0.15 of the data
    :param donors: processed OTUS
    :param mapping: mapping file
    :param a_div: tag
    :return: X_train, X_test, y_train, y_test, d_train, d_test
    """
    X = defaultdict(list)  # list of 2D otus
    d = defaultdict(list)  # list of days after transplant to be added to the learning after the flatten of the CNN
    y = defaultdict(list)  # tag (shannon a_div)
    index = defaultdict(list)
    mice_name = defaultdict(list)
    alpha_d = defaultdict(list)
    meta_d = defaultdict(list)
    for a_di in a_div.iterrows():
        try:
            donor_otu = donors.loc[mapping[a_di[0]]['Donor']]
            day = int(mapping[a_di[0]]['Day'])

            # if there are reps in the otus, we take only the first
            if len(donor_otu.shape) > 1:
                name = donor_otu.iloc[0].name
            else:
                name = donor_otu.name

            # load from the directory of the 2D microbiome according to the donor id
            X[mapping[a_di[0]]["Experiment"]].append(
                np.load(f"../{path}/{name}.npy", allow_pickle=True))  # dendogram
            # np.load(f"../2D_otus/{name}.npy", allow_pickle=True)) # pmtr
            # np.load(f"../2D_otus_ieee/{name}.npy", allow_pickle=True)) # ieee
            d[mapping[a_di[0]]["Experiment"]].append(day)
            # y[mapping[a_di[0]]["Experiment"]].append(a_di[1][specie_name])
            y[mapping[a_di[0]]["Experiment"]].append(a_di[1][0])
            index[mapping[a_di[0]]["Experiment"]].append(mapping[a_di[0]]["Experiment"])
            mice_name[mapping[a_di[0]]["Experiment"]].append(mapping[a_di[0]]["mice_ID"])
            if alpha_donors is not None:
                try:
                    donor_alpha = alpha_donors.loc[mapping[a_di[0]]['Donor']]
                except:
                    donor_alpha = alpha_donors.loc[int(mapping[a_di[0]]['Donor'])]
                alpha_d[mapping[a_di[0]]["Experiment"]].append(donor_alpha.values.item())

            if meta_ is not None:
                try:
                    meta = meta_.loc[mapping[a_di[0]]['Donor']]
                except:
                    meta = meta_.loc[[i for i in meta_.index if mapping[a_di[0]]['Donor'][:-2] in i][0]]
                meta_d[mapping[a_di[0]]["Experiment"]].append(list(meta.values))

        except KeyError:
            pass

    if test_size == 0:
        if alpha_donors is None:
            V = [None, None, None]
        else:
            V = [None, None, None, None]
        for k in X.keys():
            V_1 = X[k], y[k], d[k], alpha_d[k]
            for i in range(len(V)):
                if V[i] is None:
                    V[i] = np.array(V_1[i])
                else:
                    V[i] = np.append(V[i], np.array(V_1[i]), axis=0)
        return V

    # train test split
    V = [None, None, None, None, None, None, None, None]
    if alpha_donors is not None:
        V.extend([None, None])
    if meta_ is not None:
        V.extend([None, None])

    for k in X.keys():
        if not donors_grouped:
            if alpha_donors is None:
                V_1 = train_test_split(X[k], y[k], d[k], index[k], test_size=test_size)
            else:
                V_1 = train_test_split(X[k], y[k], d[k], index[k], alpha_d[k], test_size=test_size)
            # no need for index test, but we use it for dummy mice_group_train
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
            sp = [i for i in gss.split(X[k], groups=mice_name[k])]
            train_idx = sp[0][0]
            test_idx = sp[0][1]
            X_train, X_test = np.array(X[k])[train_idx], np.array(X[k])[test_idx]
            y_train, y_test = np.array(y[k])[train_idx], np.array(y[k])[test_idx]
            d_train, d_test = np.array(d[k])[train_idx], np.array(d[k])[test_idx]
            if alpha_donors is not None:
                alpha_d_train, alpha_d_test = np.array(alpha_d[k])[train_idx], np.array(alpha_d[k])[test_idx]
                alpha_d_train, alpha_d_test = alpha_d_train.astype(np.float32), alpha_d_test.astype(np.float32)

            if meta_ is not None:
                meta_d_train, meta_d_test = np.array(meta_d[k])[train_idx], np.array(meta_d[k])[test_idx]
                meta_d_train, meta_d_test = meta_d_train.astype(np.float32), meta_d_test.astype(np.float32)

            idx_train = np.array(index[k])[train_idx]
            mice_group_train = np.array(mice_name[k])[train_idx]

            if alpha_donors is None:
                V_1 = (X_train, X_test, y_train, y_test, d_train, d_test, idx_train, mice_group_train)
            else:
                if meta_ is None:
                    V_1 = (X_train, X_test, y_train, y_test, d_train, d_test, alpha_d_train, alpha_d_test, idx_train,
                           mice_group_train)
                else:
                    V_1 = (X_train, X_test, y_train, y_test, d_train, d_test, alpha_d_train, alpha_d_test, meta_d_train,
                           meta_d_test, idx_train,
                           mice_group_train)
        for i in range(len(V)):
            if V[i] is None:
                V[i] = np.array(V_1[i])
            else:
                V[i] = np.append(V[i], np.array(V_1[i]), axis=0)
    if alpha_donors is None:
        X_train, X_test, y_train, y_test, d_train, d_test, index_train, mice_group_train = V
    else:
        if meta_ is None:
            X_train, X_test, y_train, y_test, d_train, d_test, alpha_d_train, alpha_d_test, index_train, mice_group_train = V
        else:
            X_train, X_test, y_train, y_test, d_train, d_test, alpha_d_train, alpha_d_test, meta_d_train, meta_d_test, index_train, mice_group_train = V

    if augmented_data is not None:
        fabX, fabShannon, days = augmented_data
        pre = otu22d(fabX, save=False)
        fabX = dendogram_ordering(pre, fabX, save=False)[0]
        X_train = np.append(X_train, fabX, axis=0)
        y_train = np.append(y_train, fabShannon.values)
        d_train = np.append(d_train, days)
        index_train = np.append(index_train, np.array(["fab" for i in fabShannon]))
        mice_group_train = np.append(mice_group_train, np.array(["fab" for i in fabShannon]))
    if not donors_grouped:
        mice_group_train = None
    if alpha_donors is None:
        return X_train, X_test, y_train, y_test, d_train, d_test, index_train, mice_group_train
    else:
        if meta_ is None:
            return X_train, X_test, y_train, y_test, d_train, d_test, index_train, mice_group_train, alpha_d_train, alpha_d_test
        else:
            return X_train, X_test, y_train, y_test, d_train, d_test, index_train, mice_group_train, alpha_d_train, alpha_d_test, meta_d_train, meta_d_test


def load_data_train_test_valid(org_X_train, X_test, org_y_train, y_test, org_d_train, d_test, index_train,
                               donor_group_train=None, alpha_donors_train=None, alpha_donors_test=None, meta_train=None,
                               meta_test=None):
    """
    a. split the train we got from load_data_2d_train_test or from load_data_1d_train_test to
    train and validation without seed. (15% of the whole data as validation)
    b. transform the train, validation and test to tensors
    :param X_train: contains 2D otus
    :param X_test: contains 2D otus (15% of the data)
    :param y_train: a_divs of the train
    :param y_test: a_divs of the test
    :param d_train: days after transplant
    :param d_test: days after transplant
    :return: tensors of train test and validation (x and y)
    """
    fab_dataset = None
    V = [None, None, None, None, None, None]
    if alpha_donors_train is not None:
        V.extend([None, None])
    if meta_train is not None:
        V.extend([None, None])
    for expe in np.unique(index_train):
        x = org_X_train[index_train == expe]
        y = org_y_train[index_train == expe]
        d = org_d_train[index_train == expe]
        if alpha_donors_train is not None:
            a = alpha_donors_train[index_train == expe]
        if meta_train is not None:
            m = meta_train[index_train == expe]

        if expe != "fab":
            if donor_group_train is None:
                if alpha_donors_train is None:
                    V_1 = train_test_split(x, y, d, test_size=0.175)  # 0.175 of train means 0.15 of the whole data
                else:
                    if meta_ is None:
                        V_1 = train_test_split(x, y, d, a,
                                               test_size=0.175)  # 0.175 of train means 0.15 of the whole data
                    else:
                        V_1 = train_test_split(x, y, d, a, m,
                                               test_size=0.175)  # 0.175 of train means 0.15 of the whole data
            else:
                dgt = donor_group_train[index_train == expe]
                gss = GroupShuffleSplit(n_splits=1, test_size=0.175)
                sp = [i for i in gss.split(x, groups=dgt)]
                train_idx = sp[0][0]
                valid_idx = sp[0][1]
                x_t, x_v = x[train_idx], x[valid_idx]
                y_t, y_v = y[train_idx], y[valid_idx]
                d_t, d_v = d[train_idx], d[valid_idx]
                V_1 = (x_t, x_v, y_t, y_v, d_t, d_v)
                if alpha_donors_train is not None:
                    a_t, a_v = a[train_idx], a[valid_idx]
                    V_1 = (x_t, x_v, y_t, y_v, d_t, d_v, a_t, a_v)
                    if meta_train is not None:
                        m_t, m_v = m[train_idx], m[valid_idx]
                        V_1 = (x_t, x_v, y_t, y_v, d_t, d_v, a_t, a_v, m_t, m_v)

            for i in range(len(V)):
                if V[i] is None:
                    V[i] = np.array(V_1[i])
                else:
                    V[i] = np.append(V[i], np.array(V_1[i]), axis=0)
        else:
            fab_dataset = data.TensorDataset(torch.tensor(x), torch.tensor(y), torch.tensor(d))
    if alpha_donors_train is None:
        X_train, X_valid, y_train, y_valid, d_train, d_valid = V
        train_dataset = data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(d_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test), torch.tensor(d_test))
        valid_dataset = data.TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid), torch.tensor(d_valid))
    else:
        if meta_train is None:
            X_train, X_valid, y_train, y_valid, d_train, d_valid, a_train, a_valid = V
            train_dataset = data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(d_train),
                                               torch.tensor(a_train))
            test_dataset = data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test), torch.tensor(d_test),
                                              torch.tensor(alpha_donors_test))
            valid_dataset = data.TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid), torch.tensor(d_valid),
                                               torch.tensor(a_valid))
        else:
            X_train, X_valid, y_train, y_valid, d_train, d_valid, a_train, a_valid, m_train, m_valid = V
            train_dataset = data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(d_train),
                                               torch.tensor(a_train), torch.tensor(m_train))
            test_dataset = data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test), torch.tensor(d_test),
                                              torch.tensor(alpha_donors_test), torch.tensor(meta_test))
            valid_dataset = data.TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid), torch.tensor(d_valid),
                                               torch.tensor(a_valid), torch.tensor(m_valid))

    return train_dataset, valid_dataset, test_dataset, y_train, y_valid, y_test, fab_dataset


def load_data_1d_train_test(donors, mapping, a_div, donors_grouped=True):
    """
    a. load 1 dimensional microbiome data days after transplant data and tag data
    according to mapping file
    b. split train and test, while the test is 0.15 of the data
    :param donors: processed OTUs
    :param mapping:
    :param a_div: tag
    :return: X_train, X_test, y_train, y_test, d_train, d_test
    """
    X = []  # list of otus
    d = []  # list of days after transplant
    y = []  # tag (shannon a_div)
    index = []
    donor_group = []
    for a_di in a_div.iterrows():
        try:
            donor_otu = donors.loc[mapping[a_di[0]]['Donor']]
            day = int(mapping[a_di[0]]['Day'])

            # if there are reps in the otus, we take only the first
            if len(donor_otu.shape) > 1:
                x = donor_otu.iloc[0]
            else:
                x = donor_otu

            X.append(x.to_numpy())
            d.append(day)
            y.append(a_di[1][0])
            donor_group.append(mapping[a_di[0]]["mice_ID"])
            index.append(mapping[a_di[0]]["Experiment"])

        except KeyError:
            pass
    X = np.array(X)
    D = np.array(d)
    Y = np.array(y)
    donor_group = np.array(donor_group)
    index = np.array(index)

    V = [None, None, None, None, None, None, None, None]
    for expe in np.unique(index):
        x = X[index == expe]
        y = Y[index == expe]
        d = D[index == expe]
        dgt = donor_group[index == expe]
        idx = index[index == expe]

        if not donors_grouped:
            V_1 = train_test_split(x, y, d, idx, test_size=0.15)
            # index test is not needed but left as a dummy for donor_group_train
            # V_1 = V_1[:-1]
            # V_1.append(None)

        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15)
            sp = [i for i in gss.split(x, groups=dgt)]
            train_idx = sp[0][0]
            test_idx = sp[0][1]
            X_train, X_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            d_train, d_test = d[train_idx], d[test_idx]
            index_train, index_test = idx[train_idx], idx[test_idx]
            donor_group_train = dgt[train_idx]
            V_1 = X_train, X_test, y_train, y_test, d_train, d_test, index_train, donor_group_train

        for i in range(len(V)):
            if V[i] is None:
                V[i] = np.array(V_1[i])
            else:
                V[i] = np.append(V[i], np.array(V_1[i]), axis=0)

    X_train, X_test, y_train, y_test, d_train, d_test, index_train, donor_group_train = V
    if not donors_grouped:
        donor_group_train = None
    return X_train, X_test, y_train, y_test, d_train, d_test, index_train, donor_group_train


if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')
    # Set seed - test should be always the same
    # random.seed(30)
    # torch.manual_seed(30)
    # torch.cuda.manual_seed_all(30)
    # np.random.seed(30)
    dataset = sys.argv[1]
    mode = sys.argv[2]

    try:
        rank = sys.argv[5]
    except:
        rank = None

    D = None
    # div_a_otu = pd.read_csv(
    #     '../a_divs/alpha_diversity_baby_allergy_gdm_chimo_mice.csv', index_col=0)
    # donors = pd.read_csv('tax_7_rel_BGCA.csv', index_col=0)
    #
    a_div_shannon = pd.read_csv(
        '../a_divs/alpha_diversity_baby_allergy_gdm_chimo_mice_shannon.csv',
        index_col=0)
    mice = pd.read_csv("../new_task_data/tax_4_relative_sum.csv", index_col=0)
    trans_mice = pd.read_csv("../trans_data/tax_4_relative_sum_trans_mice.csv", index_col=0)
    mice.columns = [i.replace("; ", ";") for i in mice.columns]
    common = list(mice.columns.intersection(trans_mice.columns))
    name_ = common[0].split("__")[4]
    print(name_)

    species = pd.read_csv(f"../final_fig/species_learning/tax7_relative_sum.csv", index_col=0)
    species_cols = list(species.columns)
    #for i in range(len(species_cols)):
    for i in range(1):
        specie_name = species_cols[i]
        print(specie_name)
        spec = species[specie_name]
        concentrate = pd.read_csv(f"../new_task_data/tags/{name_}.csv",index_col=0)
        # concentrate = pd.read_csv("../new_task_data/Gammaproteobacteria.csv",index_col=0)
        # concentrate = pd.read_csv("../new_task_data/Lactobacillales.csv", index_col=0)
        # trate = pd.read_csv("../new_task_data/Verrucomicrobiales.csv",index_col=0)
        # concentrate = pd.read_csv("../new_task_data/Bifidobacteriales.csv",index_col=0)
        # # bgu
        # a_div_shannon = pd.read_csv(
        #     '../Proccessed_data/bgu/shannon_for_learning.csv',
        #     index_col=0)

        donors, path, input_dim, alpha_donors, meta_ = load_nni_data(dataset, mode)
        # donors = pd.read_csv("../2D_otus_dendogram_ordered/0_fixed_ordered_tax7.scv", index_col=0) # return to this
        mapping = pd.read_csv('../Raw_data/mapping_for_preprocess.csv', index_col=0)
        mapping.columns = mapping.loc['ID']
        a_div = concentrate  # a_div_shannon
        # a_div = species  # a_div_shannon
        #a_div = a_div_shannon
        # a_div = a_div.loc[a_div["0"] != 0] # only for shannon

        # if rank is not None:
        #     a_div = rank_alpha(a_div)
        #     a_div = a_div.to_frame().astype(np.float32)
        #     a_div = a_div.rename(columns={"rank": "0"})

        # plt.hist(a_div.values.squeeze(), bins=30)
        # plt.xlabel("Shannon index")
        # plt.ylabel("Frequency")
        # plt.title("Shannon alpha diversity ground truth")
        # plt.show()

        m = sys.argv[3]
        if sys.argv[4] is not None and sys.argv[4] == "random":
            augmented_data = randomAugment(donors, a_div, mapping)
        elif sys.argv[3] is not None and sys.argv[3] == "ccgan":
            augmented_data = generate_with_ccgan(donors)
        else:
            augmented_data = None

        D = nni.get_next_parameter()

        if m.lower() == "cnn1":
            model = CNN_1l
            V = load_data_2d_train_test(donors, mapping, a_div, path, augmented_data)
            if len(D.values()) == 0:
                D = {

                    "l1_loss": 0.632356554505828,
                    "weight_decay": 0.009327439593222919,
                    "lr": 0.001,
                    "batch_size": 64,
                    "activation": "elu",
                    "dropout": 0.3,
                    "linear_dim_divider_1": 4,
                    "linear_dim_divider_2": 5,
                    "kernel_size_a": 3,
                    "kernel_size_b": 17,
                    "stride": 5,
                    "channels": 14


                   

                }
        # edit configuration -> parameters
        if m.lower() == "cnn" or m.lower() == "cnn2" or m.lower() == "rankcnn" or m.lower() == "mtlcnn":
            if m.lower() == "rankcnn":
                pass
            elif "mtl" in m.lower():
                pass
            else:
                model = CNN

            # deafult 2 conv CNN parameters, if not in nni mode:
            if len(D.values()) == 0:
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
                    "linear_dim_divider_2": 11,
                    "rank": True if rank else False,

                }
            V = load_data_2d_train_test(donors, mapping, a_div, path, augmented_data,
                                        alpha_donors=
                                        alpha_donors, meta_=meta_

                                        )
        if m.lower() == "dcnn2":
            pass
            # deafult 2 conv CNN parameters, if not in nni mode:
            if len(D.values()) == 0:
                D = {

                    "l1_loss": 0.5081236409390953,
                    "weight_decay": 0.0012384649898148508,
                    "lr": 0.001,
                    "batch_size": 128,
                    "activation": "tanh",
                    "dropout": 0,
                    "kernel_size_d_a": 3,
                    "kernel_size_d_b": 1,
                    "kernel_size_a": 2,
                    "kernel_size_b": 14,
                    "dstride": 2,
                    "stride": 5,
                    "dpadding": 2,
                    "padding": 1,
                    "padding_2": 0,
                    "kernel_size_a_2": 1,
                    "kernel_size_b_2": 1,
                    "stride_2": 4,
                    "dchannels": 2,
                    "channels": 6,
                    "channels_2": 12,
                    "linear_dim_divider_1": 3,
                    "linear_dim_divider_2": 9

                }
            V = load_data_2d_train_test(donors, mapping, a_div, path, augmented_data, alpha_donors=alpha_donors)

        if m.lower() == "naeive":
            model = Naeive
            if len(D.values()) == 0:
                D = {
                    "l1_loss": 0.5901589355718209,
                    "weight_decay": 0.006268919878886442,
                    "lr": 0.001,
                    "batch_size": 256,
                    "activation": "elu",
                    "dropout": 0.05,
                    "linear_dim_1": 317,
                    "linear_dim_2": 126

                }
            V = load_data_1d_train_test(donors, mapping, a_div)
        if m.lower() == "svr":
            model = SVR
            D = {}
            V = load_data_1d_train_test(donors, mapping, a_div)

        if m.lower() == "ridge":
            model = Ridge
            D = {"alpha": 1}
            V = load_data_1d_train_test(donors, mapping, a_div)

        if m.lower() == "knn":
            model = KNeighborsRegressor
            D = {"n_neighbors": 5,
                 "algorithm": "auto",
                 "metric": "minkowski"
                 }
            V = load_data_1d_train_test(donors, mapping, a_div)

        if m.lower() == "xgboost":
            model = XGBRegressor
            D = {"booster": "gbtree",
                 "eta": 0.3,
                 "max_depth": 5,
                 "lambda": 1.5,
                 "alpha": 0.5}

        if m.lower() == "rf":
            model = RandomForestRegressor
            D = {}
            V = load_data_1d_train_test(donors, mapping, a_div)

        all_r2_tr, all_r2_te, all_c_tr, all_c_te, all_r2_tr_round, all_r2_val_round, all_r2_test, all_c_test = [], [], [], [], [], [], [], []
        for e in range(10):
            try:
                # e-cross validation (split the train to train and validation) and transform everything to tensor:
                V1 = load_data_train_test_valid(*V)  # *V send the parameters of V one by one
                if "batch_size" not in D.keys():
                    D["batch_size"] = 1
                r2_tr, r2_te, c_tr, c_te, r2_tr_round, r2_val_round, r2_test, c_test = POC(*V1, model=model, parms=D,
                                                                                           input_dim=input_dim,
                                                                                           pairwise=True if "rank" in m.lower() else False)
                # all_r2_tr_round.append(r2_tr_round)
                # all_r2_val_round.append(r2_val_round)
                all_r2_tr.append(r2_tr)
                all_r2_te.append(r2_te)
                all_c_tr.append(c_tr)
                all_c_te.append(c_te)
                all_r2_test.append(r2_test)
                all_c_test.append(c_test)
                print(f"R2 train:{r2_tr},\n"
                      f"R2 validation: {r2_te},\n"
                      f"R2 test: {r2_test},\n"
                      f"corr train: {c_tr},\n"
                      f"corr validation: {c_te},\n"
                      f"corr test: {c_test}")
            except Exception as ex:
                nni.report_final_result(-np.inf)
                raise ex
        print(f"\nSTD:")
        print(f"R2 train:{np.std(all_r2_tr)},\n"
              f"R2 validation: {np.std(all_r2_te)},\n"
              f"R2 test: {np.std(all_r2_test)},\n"
              f"corr train: {np.std(all_c_tr)},\n"
              f"corr validation: {np.std(all_c_te)},\n"
              f"corr test: {np.std(all_c_test)},\n"
              # f"R2 rounded train: {np.std(all_r2_tr_round)},\n"/
              # f"R2 rounded valid: {np.std(all_r2_val_round)}"
              )
        print(f"Mean:")
        print(f"R2 train:{np.mean(all_r2_tr)},\n"
              f"R2 validation: {np.mean(all_r2_te)},\n"
              f"R2 test: {np.mean(all_r2_test)},\n"
              f"corr train: {np.mean(all_c_tr)},\n"
              f"corr validation: {np.mean(all_c_te)}\n"
              f"corr test: {np.mean(all_c_test)}\n"
              # f"R2 rounded train: {np.mean(all_r2_tr_round)},\n"
              # f"R2 rounded valid: {np.mean(all_r2_val_round)}"
              )

        # report the mean results of e validations
        nni.report_final_result(np.mean(all_r2_te))
