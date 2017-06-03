import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import  matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
'''
feature importance
'''
def feature_importance_plot():

    model_file = open("xgb_model", 'rb')
    model = pickle.load(model_file)
    fig, ax = plt.subplots(1, 1, figsize=(15,15))
    xgb.plot_importance(model ,max_num_features=30, grid=False, height=0.5, ax=ax)
    plt.show()