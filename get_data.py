'''
pre process data.
wrote by Yu.
'''
import pandas as pd
import numpy as np

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
    df: dataframe
@:return
    df: transformed dataframe
    
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

if __name__ == "__main___":
    df = pd.read_csv("asset/train.csv")
    df = handle_non_numeric_data(df)
    print(df.head())
