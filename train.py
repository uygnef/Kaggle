'''
training
written by Yu Feng.
'''
import get_data
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import new_feature


#https://www.kaggle.com/optidatascience/use-partial-pca-for-collinearity-lb-0-328-w-xgb/notebook/notebook
#PCA did not lead to improve for xgboost.
# def PCA(df):
#     import bisect
#     internal_chars = ['full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 'num_room', 'kitch_sq', 'state']
#
#     corrmat = df[internal_chars].corr()
#     print(corrmat)
#
#     pca.fit(df)
#     varexp = pca.explained_variance_ratio_.cumsum()
#     print(varexp)
#     cutoff = bisect.bisect(varexp, 0)
#     print(cutoff)
#     return df

'''
naieivxgb:

'''
def XGB(x_train, y_train, x_test):

    xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
    }

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
                       verbose_eval=50, show_stdv=False)
    cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
    num_boost_rounds = len(cv_output)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

    file = open("xgb_model", 'wb')
    pickle.dump(model , file)

    y_predict = model.predict(dtest)
    output = pd.DataFrame({'price_doc': y_predict})
    print(output.head())
    return output

if __name__ == "__main__":
    df = pd.read_csv("asset/train.csv", skipinitialspace=True, parse_dates=['timestamp'])
    test_pd = pd.read_csv("asset/test.csv", skipinitialspace=True, parse_dates=['timestamp'])
    get_data.data_cleaning(df, test_pd)
    get_data.feature_engineering(df, test_pd)

    df = new_feature.get_new_feature(df, "train")
    test_pd = new_feature.get_new_feature(test_pd, "test")

    test_id = test_pd['id']
    test_id = test_id.to_frame()
    test = get_data.pre_process_test(test_pd)

    train_X, test_X, train_y, test_y = get_data.pre_process(df)
   # print(df.head())
    result = XGB(train_X, train_y, test)


    file = open("1", 'wb')
    pickle.dump(test_id.join(result) , file)

   # print(df.head())
