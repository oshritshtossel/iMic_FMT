import random

import pandas as pd
import numpy as np


def load_data_1d(donors, mapping, a_div):
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
            index.append(mapping[a_di[0]])

        except KeyError:
            pass
    X = np.array(X)
    d = np.array(d)
    y = np.array(y)
    index = np.array(index)

    return X, y, d, index


def randomAugment(donors, a_div, mapping):
    mapping.columns = mapping.loc['ID']

    X, Y, d, index = load_data_1d(donors, mapping, a_div)
    fabX = []  # fabricated donors
    fabShannon = []  # fabricated a_div
    days = []
    # split accord to a specific day, in order to find similarities in specific days:
    for val in np.unique(d):
        x = X[d == val]
        y = Y[d == val]
        i = index[d == val]

        # create differences matrix:
        chosen = abs(
            np.expand_dims(y, 0) - np.expand_dims(y, 0).T) < 2.5e-2  # 5e^-2 is the value chosen for a small difference
        # add names for columns and rows- mice ID, including days after transplantation
        ccc = pd.DataFrame(data=chosen, columns=i[:, 1], index=i[:, 1])

        for c, col in enumerate(ccc):
            for r, i in enumerate(ccc.index):
                if col == i:  # drop cases the the close shannon are from the same donor
                    chosen[r, c] = False
        for r, row in enumerate(chosen):
            for c, col in enumerate(chosen[r]):
                if chosen[r, c] == True:
                    shannon = (y[c] + y[r]) / 2  # generate fabricated a_div
                    x_new = []
                    for j in range(len(x[c])):
                        x_new.append(random.choice([x[c, j], x[
                            r, j]]))  # generate fabricated otus as random values of the couple with the close shannons
                    x_new = np.array(x_new)
                    days.append(val)
                    fabX.append(x_new)
                    fabShannon.append(shannon)

    fabX = np.array(fabX)
    fabShannon = pd.Series(np.array(fabShannon))
    fabX = pd.DataFrame(data=fabX, columns=donors.columns)
    return fabX, fabShannon, days

    # fabX.to_csv("augmented_data_tax7_subpcalog.csv")
    # fabShannon.to_csv("augmented_shannon_tax7_subpcalog.csv")
