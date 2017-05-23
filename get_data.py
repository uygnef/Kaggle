'''
pre process data.
written by Yu.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def load_data():
    data = {}
    data['macro'] = pd.read_csv("asset/macro.csv")
    data['test'] = pd.read_csv("asset/test.csv")
    data['train'] = pd.read_csv("asset/train.csv")
    return data


#TODO: PCA, feature selection...
def pre_process(df):
    handle_non_numeric_data(df)
    data_filling(df)

    #copy from https://www.kaggle.com/bguberfain/naive-xgb-lb-0-317
    # Add month-year



    return df_train_test_split(df)



'''
naive approach to fill data. filling empty data with mean of this column.
'''
#TODO: filling with nearest value
def data_filling(df):
    df.loc[df['state'] == 33, 'state'] = df['state'].mode().iloc[0]
    df.loc[df['build_year'] == 20052009, 'build_year'] = 2007
    for col in df:
        try:
            df[col].fillna(df[col].mean(), inplace=True)
        except TypeError:
            df[col].fillna(method='pad')
            df[col].fillna(method='bfill')


'''
naive method to transfer non numeric feature to int.
only assign each value(class) a number.
@:param
    df: data frame
@:return
    df: transformed data frame
    
'''
#TODO: find better value for each class
def handle_non_numeric_data(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(list(df[c].values))


'''
split raw data frame to test and train.
@:return
    train features, test features, train result, test result.
'''
def df_train_test_split(df, test_size = 0.2, random_state = 42):
    y = df['price_doc']
    #TODO: normalized timestamp
    df.drop(['price_doc', 'id', 'timestamp'], 1, inplace=True)

    return train_test_split(df,y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    df = pd.read_csv("asset/train.csv", skipinitialspace=True, parse_dates=['timestamp'])
    train_x, test_x, train_y, test_y = pre_process(df)
    print(len(train_x), len(test_x), len(train_y), len(test_y))
    print(df.head())
    #check if has Nan value
    for i in df:
        if df[i].isnull().sum() != 0:
            print(i, df[i].isnull().sum())

