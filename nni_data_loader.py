import pandas as pd
import os



def load_nni_data(name_of_dataset, D_mode):
    """
    load otu table, tag, map and determine the task of each dataset
    :param name_of_dataset: one of :
    "BGU_T0", "BGU", "Allergy_MILK_T0", "GDM_T2_T3", "IBD", "Gastro_vs_oral"
    :param D_mode: one of : "1D", "IEEE", "dendogram"
                   it impacts the order of the otus, if its sub-pca log or relative sum and on the path.

    :param tag_name: its defualt is None, when there is only 1 tag,
                     in "BGU_T0" or "BGU": is one of:
                     "dsc", "ssc", "vat", "li"
                     in "IBD": is one of:
                     "IBD", "CD", "UC"
    :param with_map: can get True or False
                     True - means using the features in mapping file in addition to microbiome
                     False - means using only the microbiome
    :return: otu, tag,map, input_dim, task
    """
    org_path = os.getcwd()

    if "\\" in os.getcwd():
        while os.getcwd().split("\\")[-1] != "Thesis":
            os.chdir("..")
    else:
        while os.getcwd().split("/")[-1] != "Thesis":
            os.chdir("..")

    path_of_2D_matrix = None
    biomarkers = None
    group = None

    if name_of_dataset == "subpca":
        if D_mode == "IEEE":
            otu = pd.read_csv("tax_encoding/tax_7_rel_BGCA.csv", index_col=0)
            path_of_2D_matrix = "2D_otus_ieee"
            input_dim = (8, len(otu.columns))
        elif D_mode == "1D":
            otu = pd.read_csv('tax_encoding/tax7.csv', index_col=0)
            path_of_2D_matrix = None
            input_dim = len(otu.columns) + 1

        else:
            otu = pd.read_csv("2D_otus_dendogram_ordered/0_fixed_ordered_tax7.scv",
                              index_col=0)
            path_of_2D_matrix = "2D_otus_dendogram_ordered"
            input_dim = (8, len(otu.columns))

    if name_of_dataset == "mean":
        if D_mode == "IEEE":
            otu = pd.read_csv("Data/BGU_otu_sum_relative_rare_bact_5_tax_7.csv", index_col=0)
            path_of_2D_matrix = "2D_otus_IEEE_BGU"
            input_dim = (8, len(otu.columns))
        elif D_mode == "1D":
            otu = pd.read_csv('tax_encoding/tax_7_log_mean.csv', index_col=0)
            path_of_2D_matrix = None
            input_dim = len(otu.columns) + 1
            alpha_donors = None

        else:
            otu = pd.read_csv("2D_otus_dendogram_ordered_log_mean/0_fixed_ordered_tax7.csv",
                              index_col=0)
            path_of_2D_matrix = "2D_otus_dendogram_ordered_log_mean"
            input_dim = (8, len(otu.columns))
            alpha_donors_raw = pd.read_csv("a_divs/tax_7_relative_BGCA_adiv_shannon.csv", index_col=0)
            # remove donors with 0 a_div
            alpha_donors_raw = alpha_donors_raw.loc[alpha_donors_raw["0"] != 0]
            # remove reptitions (leave only the last sample for each subject)
            no_duplicates = dict()
            for donor, donor_df in alpha_donors_raw.groupby(alpha_donors_raw.index):
                no_duplicates[donor] = donor_df.iloc[-1].values.item()

            alpha_donors = pd.Series(data=no_duplicates)
            alpha_donors = alpha_donors.to_frame()

            meta_ =pd.read_csv("final_fig/Fig_2/metadata.csv",index_col=0)

    if name_of_dataset == "corr":
        if D_mode == "IEEE":
            pass
        elif D_mode == "1D":
            otu = pd.read_csv('tax_encoding/log_mean_tax_7_corr.csv', index_col=0)
            path_of_2D_matrix = None
            input_dim = len(otu.columns) + 1
        else:
            otu = pd.read_csv("donors_mice_corr_2d/0_fixed_ordered_tax7.csv",
                              index_col=0)
            path_of_2D_matrix = "donors_mice_corr_2d"
            input_dim = (7, len(otu.columns))

    if name_of_dataset == "BGU":
        if D_mode == "IEEE":
            pass
        if D_mode == "dendogram":
            otu = pd.read_csv("bgu_2d_dendogram/0_fixed_ordered_tax7.csv",
                              index_col=0)
            path_of_2D_matrix = "bgu_2d_dendogram"
            input_dim = (7, len(otu.columns))

    os.chdir(org_path)
    return otu, path_of_2D_matrix, input_dim, alpha_donors,meta_
