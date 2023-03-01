from collections import defaultdict
import pandas as pd
import numpy as np
import shap
import torch
from scipy.stats import rv_histogram, spearmanr
import matplotlib as mpl
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from CNN_model.CNN2convlayer import CNN
from microbiome2matrix import otu22d, dendogram_ordering



def end_to_end_model(donors,i,no_abx=True,meta_check=False):
    name = i
    mapping = pd.read_csv("human_data/mapping.csv", index_col=0)
    a_div_shannon = pd.read_csv("human_data/a_div_rep.csv", index_col=0)
    #a_div_shannon = pd.read_csv("human_data/order_rep.csv", index_col=0)
    if i != None:
        orders = a_div_shannon.columns
        order = orders[i]
        name = order.split("__")[-1]
        print(order)
        a_div_shannon= a_div_shannon[order].to_frame()

    alpha_donors = pd.read_csv("human_data/a_div_donors.csv", index_col=0)

    if no_abx:
        M1 = mapping.T[mapping.T["Experiment"] == "PRJDB4959"]
        M2 = mapping.T[mapping.T["Experiment"] == "PRJNA428898"]
        no_abx_m = pd.concat([M1, M2])
        common_shannon = list(set(no_abx_m["ID"].values).intersection(a_div_shannon.index))
        common_a_donors = list(set(no_abx_m["Donor"].values).intersection(alpha_donors.index))
        common_donors = list(set(no_abx_m["Donor"].values).intersection(donors.index))
        donors = donors.loc[common_donors]
        alpha_donors = alpha_donors.loc[common_a_donors]
        a_div_shannon = a_div_shannon.loc[common_shannon]
        mapping = no_abx_m.T

    else:
        M1 = mapping.T[mapping.T["Experiment"] == "ERP021216"]
        M2 = mapping.T[mapping.T["Experiment"] == "PRJNA221789"]
        M3 = mapping.T[mapping.T["Experiment"] == "PRJNA238042"]
        M4 = mapping.T[mapping.T["Experiment"] == "PRJNA238486"]
        M5 = mapping.T[mapping.T["Experiment"] == "PRJNA380944"]
        M6 = mapping.T[mapping.T["Experiment"] == "PRJNA412501"]

        no_abx_m = pd.concat([M1,M2,M3,M4,M5,M6])
        if meta_check:
            no_abx_m = M1

        common_shannon = list(set(no_abx_m["ID"].values).intersection(a_div_shannon.index))
        common_a_donors = list(set(no_abx_m["Donor"].values).intersection(alpha_donors.index))
        common_donors = list(set(no_abx_m["Donor"].values).intersection(donors.index))
        donors = donors.loc[common_donors]
        alpha_donors = alpha_donors.loc[common_a_donors]
        a_div_shannon = a_div_shannon.loc[common_shannon]
        mapping = no_abx_m.T

    mapping.columns = mapping.loc['ID']
    model = load_model((8, 178),name)
    X_train, X_test, y_train, y_test, d_train, d_test, index_train, mice_group_train, alpha_d_train, alpha_d_test,meta_d_train, meta_d_test = load_data_2d_train_test(
        donors, mapping, a_div_shannon,
        "2d_human_data",
        alpha_donors=alpha_donors,meta_=mapping)

    # merge train and test
    alpha_d_test = np.concatenate((alpha_d_train, alpha_d_test), axis=0)
    y_test = np.concatenate((y_train, y_test), axis=0)
    d_test = np.concatenate((d_train, d_test), axis=0)
    X_test = np.concatenate((X_train, X_test), axis=0)
    meta_d_test = np.concatenate((meta_d_train, meta_d_test), axis=0)

    otus_tensor = torch.tensor(X_test)
    # otus_tensor_test = torch.tensor(X_test)
    days = torch.tensor(d_test)
    a_donors = torch.tensor(alpha_d_test)
    days = torch.stack([days, a_donors.flatten()])

    pred = model(otus_tensor, days).detach().flatten().numpy()
    print(f"SCC: {spearmanr(y_test, pred)}")
    print(f"R2 score: {r2_score(y_test,pred)}")
    return pred, y_test,meta_d_test


def load_model(shape,name):
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
    if name == None:
        model = model.load_from_checkpoint("weights/shannon_weights.ckpt", in_dim=shape, params=D)
    else:
        model = model.load_from_checkpoint(f"weights/human_data/{name}.ckpt", in_dim=shape, params=D)
    return model


def load_data_2d_train_test(donors, mapping, a_div, path, donors_grouped=True, test_size=0.175,
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

            X[mapping[a_di[0]]["Experiment"]].append(
                np.load(f"{path}/{name}.npy", allow_pickle=True))  # dendogram
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
                    meta = meta_.loc["ID"]
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
                #meta_d_train, meta_d_test = meta_d_train.astype(np.float32), meta_d_test.astype(np.float32)
                meta_d_train, meta_d_test = meta_d_train, meta_d_test

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

    if not donors_grouped:
        mice_group_train = None
    if alpha_donors is None:
        return X_train, X_test, y_train, y_test, d_train, d_test, index_train, mice_group_train
    else:
        if meta_ is None:
            return X_train, X_test, y_train, y_test, d_train, d_test, index_train, mice_group_train, alpha_d_train, alpha_d_test
        else:
            return X_train, X_test, y_train, y_test, d_train, d_test, index_train, mice_group_train, alpha_d_train, alpha_d_test, meta_d_train, meta_d_test


class AAA():
    def __init__(self, model, preprocess_func):
        self.model = model
        self.preprocess_func = preprocess_func

    def __call__(self, x):
        x,d = self.preprocess_func(x)
        return self.model(x,d).detach().flatten().numpy()


def calculate_shannon(x):
    result = 10 ** x  # cancel the initial log
    pi = result.div(result.sum(axis=1), axis=0)
    a_div = (pi * np.log(pi) * (-1)).sum(axis=1)
    return a_div


def pp(x):
    alpha_donors = calculate_shannon(x)
    alpha_donors = alpha_donors.to_numpy()
    a_donors = torch.tensor(alpha_donors)
    otus2d, names = otu22d(x, with_names=True, save=False)
    otus2d, dendogramed_df = dendogram_ordering(otus2d, x, names, save=False)
    days = torch.ones(otus2d.shape[0]) * 7
    otus_tensor = torch.tensor(otus2d)
    days = torch.stack([days, a_donors.flatten()])
    return otus_tensor, days


def main():
    i = None
    donors = pd.read_csv('human_data/donors.csv', index_col=0)
    pred, y_test,meta = end_to_end_model(donors,i,no_abx=False,meta_check=False)


if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')
    main()
    c=0