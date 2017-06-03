import pickle
import pandas
import  matplotlib.pyplot as plt
import xgboost as xgb


'''
feature importance
'''
def feature_importance_plot():

    model_file = open("xgb_model", 'rb')
    model = pickle.load(model_file)
    fig, ax = plt.subplots(1, 1, figsize=(15,15))
    xgb.plot_importance(model ,max_num_features=30, grid=False, height=0.5, ax=ax)
    plt.show()

'''
load data from "1" file.
'''
file = open("1", 'rb')
a = pickle.load(file)
a.to_csv("result.csv", index=False)

feature_importance_plot()

