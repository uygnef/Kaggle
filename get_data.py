'''
pre process data.
wrote by Yu.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    data = {}
    data['macro'] = pd.read_csv("asset/macro.csv")
    data['test'] = pd.read_csv("asset/test.csv")
    data['train'] = pd.read_csv("asset/train.csv")
    return data


'''
naive method to transfer non numeric feature to int.
only assign each value(class) a number.
@:param
    df: data frame
@:return
    df: transformed data frame
    
'''
def handle_non_numeric_data(df):
    columns = df.columns

    for column in columns:
        text_digits_vals = {}
        def convert_to_int(val):
            return text_digits_vals[val]

        if df[column].dtype not in (np.int64, np.float64):
            if column == 'timestamp':
                df[column] = pd.to_datetime(pd.Series(df[column]))
                continue
            column_contents = df[column].tolist()
            unique_elements = set(column_contents)
            x = 0
            #assign a label to element
            for unique in unique_elements:
                if unique not in text_digits_vals:
                    text_digits_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))
    return df


'''
split raw data frame to test and train.
@:return
    train features, test features, train result, test result.
'''
def dp_train_test_split(df, test_size = 0.2, random_state = 42):
    y = df['price_doc']
    df.drop('price_doc', 1, inplace=True)
    return train_test_split(df,y, test_size=test_size, random_state=random_state)


if __name__ == "__main___":
    df = pd.read_csv("asset/train.csv", skipinitialspace=True)
    df = handle_non_numeric_data(df)
    print(df.head())
    train_x, test_x, train_y, test_y = dp_train_test_split(df)
    print(len(train_x), len(test_x), len(train_y), len(test_y))
