'''
training
written by Yu Feng.
'''
import get_data
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA as sk_PCA

#https://www.kaggle.com/optidatascience/use-partial-pca-for-collinearity-lb-0-328-w-xgb/notebook/notebook
#TODO: complete PCA.
def PCA(df):
    import bisect
    pca = sk_PCA()
    pca.fit(df)
    varexp = pca.explained_variance_ratio_.cumsum()
    print(varexp)
    cutoff = bisect.bisect(varexp, 0)
    print(cutoff)
    return df

if __name__ == "__main__":
    df = pd.read_csv("asset/train.csv", skipinitialspace=True)
    get_data.pre_process(df)
   # print(df.head())
    df = PCA(df)
   # print(df.head())
