import pandas as pd
def merge_datasets(list_dfs):
    merge = pd.concat(list_dfs)
    merge = merge.fillna(0.0)
    return merge




