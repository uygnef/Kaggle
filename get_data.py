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

    # Add month-year
    return df_train_test_split(df)

def pre_process_test(df):
    handle_non_numeric_data(df)
    data_filling(df)
    df.drop(['id', 'timestamp'], 1, inplace=True)
    return df


'''
naive approach to fill data. filling empty data with mean of this column.
'''
#TODO: filling with nearest value
def data_filling(df):
    df.loc[df['state'] == 33, 'state'] = df['state'].mode().iloc[0]
    df.loc[df['build_year'] == 20052009, 'build_year'] = 2007





        # for col in df:
    #     try:
    #         df[col].fillna(df[col].mean(), inplace=True)
    #     except TypeError:
    #         df[col].fillna(method='pad')
    #         df[col].fillna(method='bfill')


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


def data_cleaning(train, test):
    # clean data
    bad_index = train[train.life_sq > train.full_sq].index
    train.ix[bad_index, "life_sq"] = np.NaN
    equal_index = [601, 1896, 2791]
    test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]
    bad_index = test[test.life_sq > test.full_sq].index
    test.ix[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.life_sq < 5].index
    train.ix[bad_index, "life_sq"] = np.NaN
    bad_index = test[test.life_sq < 5].index
    test.ix[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.full_sq < 5].index
    train.ix[bad_index, "full_sq"] = np.NaN
    bad_index = test[test.full_sq < 5].index
    test.ix[bad_index, "full_sq"] = np.NaN
    kitch_is_build_year = [13117]
    train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]
    bad_index = train[train.kitch_sq >= train.life_sq].index
    train.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[test.kitch_sq >= test.life_sq].index
    test.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
    train.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
    test.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
    train.ix[bad_index, "full_sq"] = np.NaN
    bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
    test.ix[bad_index, "full_sq"] = np.NaN
    bad_index = train[train.life_sq > 300].index
    train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
    bad_index = test[test.life_sq > 200].index
    test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
    train.product_type.value_counts(normalize=True)
    test.product_type.value_counts(normalize=True)
    bad_index = train[train.build_year < 1500].index
    train.ix[bad_index, "build_year"] = np.NaN
    bad_index = test[test.build_year < 1500].index
    test.ix[bad_index, "build_year"] = np.NaN
    bad_index = train[train.num_room == 0].index
    train.ix[bad_index, "num_room"] = np.NaN
    bad_index = test[test.num_room == 0].index
    test.ix[bad_index, "num_room"] = np.NaN
    bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
    train.ix[bad_index, "num_room"] = np.NaN
    bad_index = [3174, 7313]
    test.ix[bad_index, "num_room"] = np.NaN
    bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
    train.ix[bad_index, ["max_floor", "floor"]] = np.NaN
    bad_index = train[train.floor == 0].index
    train.ix[bad_index, "floor"] = np.NaN
    bad_index = train[train.max_floor == 0].index
    train.ix[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.max_floor == 0].index
    test.ix[bad_index, "max_floor"] = np.NaN
    bad_index = train[train.floor > train.max_floor].index
    train.ix[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.floor > test.max_floor].index
    test.ix[bad_index, "max_floor"] = np.NaN
    train.floor.describe(percentiles=[0.9999])
    bad_index = [23584]
    train.ix[bad_index, "floor"] = np.NaN
    train.material.value_counts()
    test.material.value_counts()
    train.state.value_counts()
    bad_index = train[train.state == 33].index
    train.ix[bad_index, "state"] = np.NaN
    test.state.value_counts()

    # brings error down a lot by removing extreme price per sqm
    train.loc[train.full_sq == 0, 'full_sq'] = 50
    train = train[train.price_doc / train.full_sq <= 600000]
    train = train[train.price_doc / train.full_sq >= 10000]


def feature_engineering(train, test):
    month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    train['month_year_cnt'] = month_year.map(month_year_cnt_map)

    month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    test['month_year_cnt'] = month_year.map(month_year_cnt_map)

    # Add week-year count
    week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    train['week_year_cnt'] = week_year.map(week_year_cnt_map)

    week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    test['week_year_cnt'] = week_year.map(week_year_cnt_map)

    # Add month and day-of-week
    train['month'] = train.timestamp.dt.month
    train['dow'] = train.timestamp.dt.dayofweek

    test['month'] = test.timestamp.dt.month
    test['dow'] = test.timestamp.dt.dayofweek

    # Other feature engineering
    train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
    train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)

    test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
    test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

    train.apartment_name = train.sub_area + train['metro_km_avto'].astype(str)
    test.apartment_name = test.sub_area + train['metro_km_avto'].astype(str)

    train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
    test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

    y_train = train["price_doc"] * .968 + 10
    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
    x_test = test.drop(["id", "timestamp"], axis=1)

    for c in x_train.columns:
        if x_train[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_train[c].values))
            x_train[c] = lbl.transform(list(x_train[c].values))
            # x_train.drop(c,axis=1,inplace=True)

    for c in x_test.columns:
        if x_test[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_test[c].values))
            x_test[c] = lbl.transform(list(x_test[c].values))


if __name__ == "__main__":
    df = pd.read_csv("asset/train.csv", skipinitialspace=True, parse_dates=['timestamp'])
    train_x, test_x, train_y, test_y = pre_process(df)
    print(len(train_x), len(test_x), len(train_y), len(test_y))
    print(df.head())
    #check if has Nan value
    for i in df:
        if df[i].isnull().sum() != 0:
            print(i, df[i].isnull().sum())

