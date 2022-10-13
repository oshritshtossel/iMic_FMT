from collections import defaultdict

import igraph
import numpy as np
import pandas as pd
from plotly.figure_factory._dendrogram import sch

from textreeCreate import create_tax_tree


def get_map_ieee(tree: igraph.Graph, permute=-1):
    """

    :param tree:
    :param permute:
    :return:
    """
    # self.set_height()
    # self.set_width()
    order, layers, _ = tree.bfs(0)
    # the biggest layer len of the phylogenetic tree
    width = max(layers[-1] - layers[-2], layers[-2] - layers[-3])
    # number of layers in phylogenetic tree
    height = len(layers)
    m = np.zeros((height, width))

    layers_ = defaultdict(list)
    k = 0
    for index, i in enumerate(order):
        if index != layers[k]:
            layers_[k].append(i)
        else:
            k += 1
            layers_[k].append(i)

    i = 0
    for l in layers_:
        j = 0
        for n in layers_[l]:
            m[i][j] = tree.vs.find(n)["value"]
            j = j + 1
        i += 1
    return np.delete(m, 8, 0)


def find_root(G, child):
    parent = list(G.predecessors(child))
    if len(parent) == 0:
        print(f"found root: {child}")
        return child
    else:
        return find_root(G, parent[0])


def dfs_rec(tree, node, added, depth, m):
    added[node] = True
    neighbors = tree.neighbors(node)
    sum = 0
    num_of_sons = 0
    num_of_descendants = 0
    for neighbor in neighbors:
        if added[neighbor] == True:
            continue
        val, m, descendants = dfs_rec(tree, neighbor, added, depth + 1, m)
        sum += val
        num_of_sons += 1
        num_of_descendants += descendants

    if num_of_sons == 0:
        value = tree.vs[node]["_nx_name"][1]  # the value
        n = np.zeros((len(m), 1)) / 0
        n[depth, 0] = value
        m = np.append(m, n, axis=1)
        return value, m, 1
    avg = sum / num_of_sons
    for j in range(num_of_descendants):
        m[depth][len(m.T) - 1 - j] = avg

    return avg, m, num_of_descendants


def dfs_(tree, m):
    nv = tree.vcount()
    added = [False for v in range(nv)]
    _, m, _ = dfs_rec(tree, 0, added, 0, m)
    return np.nan_to_num(m, 0)


def get_map(tree, nettree):
    order, layers, ance = tree.bfs(0)
    width = max(layers[-1] - layers[-2], layers[-2] - layers[-3])
    height = len(layers) - 1
    m = np.zeros((height, 0))
    m = dfs_(tree, m)

    # layers_ = defaultdict(list)
    # k = 0
    # for index, i in enumerate(order):
    #     if index != layers[k]:
    #         layers_[k].append(i)
    #     else:
    #         k += 1
    #         layers_[k].append(i)
    # j = 0
    # for node in layers_[k]:
    #     if node not in order:
    #         continue
    #     father = node
    #     depth=k
    #     while (True):
    #         indices = [i for i, x in enumerate(ance) if x == father]
    #         if len(indices) != 0:
    #             break
    #         father = ance[order.index(father)]
    #         depth-=1
    #     child = [order[i] for i in indices]
    #     child_values = [tree.vs[i]["_nx_name"][1] for i in child]
    #     for c in child:
    #         ance.pop(order.index(c))
    #         order.remove(c)
    #     for cv in child_values:
    #         m[depth][j] = cv
    #         m[depth - 1][j] = np.mean(child_values)
    #         j += 1
    #
    #     x = 3

    return m


def otu22d(df, save=False, with_names=False):
    M = []
    for subj in df.iloc:
        nettree = create_tax_tree(subj)
        tree = igraph.Graph.from_networkx(nettree)
        # m = get_map_ieee(tree, nettree)

        m = get_map(tree, nettree)
        M.append(m)
        # plt.imshow(m)
        # plt.show()
        if save:
            m.dump(f"2D_otus/{subj.name}.npy")
    if with_names:
        return np.array(M), ["; ".join(j[0]) for j in [x for x in nettree.nodes() if nettree.degree(x) == 1]]
    return np.array(M)


def ppp(p: np.ndarray):
    ret = []
    p = p.astype(float)
    while p.min() < np.inf:
        m = p.argmin()
        ret.append(m)
        p[m] = np.inf

    return ret


def rec(otu, bacteria_names_order):
    first_row = None
    for i in range(otu.shape[1]):
        if 2 < len(np.unique(otu[0, i, :])):
            first_row = i
            break
    if first_row is None:
        return
    X = otu[:, first_row, :]

    Y = sch.linkage(X.T)
    Z1 = sch.dendrogram(Y, orientation='left')
    idx = Z1['leaves']
    otu[:, :, :] = otu[:, :, idx]
    bacteria_names_order[:] = bacteria_names_order[idx]

    if first_row == (otu.shape[1] - 1):
        return

    unique_index = sorted(np.unique(otu[:, first_row, :][0], return_index=True)[1])

    S = []
    for i in range(len(unique_index) - 1):
        S.append((otu[:, first_row:, unique_index[i]:unique_index[i + 1]],
                  bacteria_names_order[unique_index[i]:unique_index[i + 1]]))
    S.append((otu[:, first_row:, unique_index[-1]:], bacteria_names_order[unique_index[-1]:]))

    for s in S:
        rec(s[0], s[1])


def dendogram_ordering(otu, df=None, names=None, save=True):
    if df is None and names is None:
        raise AttributeError("df or names must be given")
    if df is not None and names is None:
        names = np.array(list(df.columns))
        rec(otu, names)
        df = df[names]
    else:
        org_names = list(df.columns)
        oon = [i.replace("k__", "").replace("p__", "").replace("c__", "").replace("o__", "").replace("f__", "").replace(
            "g__", "").replace("s__", "").replace(";;", "").strip(" ;").split(";") for i in org_names]
        oon = ["; ".join([j.replace(" ", "") for j in i]) for i in oon]
        ff = []
        seen = set()
        for b, o in zip(oon, org_names):
            for b1 in names:
                if b1 == b and b1 not in seen:
                    ff.append(o)
                    seen.add(b1)
                    break
        names = ff
        df = df[names]

    if save:
        #df.to_csv("2D_otus_dendogram_ordered_log_mean/0_fixed_ordered_tax7.csv")
        #df.to_csv("bgu_2d_dendogram/0_fixed_ordered_tax7.csv")
        #df.to_csv("bgu_2d_corr_images/0_fixed_ordered_tax7.csv")
        #df.to_csv("donors_mice_corr_2d/0_fixed_ordered_tax7.csv")
        df.to_csv("2d_all_bgu_mice_union/0_fixed_ordered_tax7.csv")
        # df.to_csv("2D_otus_dendogram_ordered/0_fixed_ordered_tax7.scv") # log subpca zscore
        for m, index in zip(otu, df.index):
            #m.dump(f"2D_otus_dendogram_ordered_log_mean/{index}.npy")
            #m.dump(f"bgu_2d_dendogram/{index}.npy")
            #m.dump(f"bgu_2d_corr_images/{index}.npy")
            #m.dump(f"donors_mice_corr_2d/{index}.npy")
            m.dump(f"2d_all_bgu_mice_union/{index}.npy")
    return otu, df


if __name__ == '__main__':
    # df = pd.read_csv("NNI_experiments/tax_7_rel_BGCA.csv", index_col=0)
    # df = pd.read_csv("tax_encoding/tax7.csv", index_col=0) ###log subpca zscore
    #df = pd.read_csv("tax_encoding/tax_7_log_mean.csv", index_col=0)  ### log mean no zscore
    #df = pd.read_csv("Proccessed_data/bgu/bgu_otu_tax7_log_mean_rare_5.csv",index_col=0) #bgu
    #df = pd.read_csv("Proccessed_data/bgu/log_mean_tax7_corr_donors.csv",index_col=0)#bgu only corr
    #df = pd.read_csv("tax_encoding/log_mean_tax_7_corr.csv",index_col=0)#donors mice only corr
    df = pd.read_csv("tax_encoding/all_bgu_mice_union_bact_log_mean_tax_7.csv",index_col=0)


    # maps = []
    # i = 0
    otus2d, names = otu22d(df, with_names=True, save=False)
    dendogram_ordering(otus2d, df, names)

    #
    # start_of_new_vals_in_last_row = np.unique(otus2d[0, i - 1, :], return_index=True)[1]
    # start_of_new_vals_in_last_row = np.append(start_of_new_vals_in_last_row, otus2d.shape[2] - 1)
    #
    # for i in range(len(start_of_new_vals_in_last_row)):
    #     j = 1
    #
    #     block = otus2d[:, :, start_of_new_vals_in_last_row[i]:start_of_new_vals_in_last_row[i + 1]]
    #     first_block_row = block[:, first_row + j, :]
    #     while len(np.unique(first_block_row[0, :])) < 2:
    #         j += 1
    #         first_block_row = block[:, first_row + j, :]
    #     ac = AgglomerativeClustering(len(np.unique(first_block_row[0, :])))
    #     p = ac.fit_predict(first_block_row.T)
    #     indexes = ppp(p)
    #     block = block[:, :, indexes]
    #     otus2d[:, :, start_of_new_vals_in_last_row[i]:start_of_new_vals_in_last_row[i + 1]] = block
