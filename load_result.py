import pickle
import pandas
import  matplotlib.pyplot as plt
import xgboost as xgb

'''
load data from "1" file.
'''
file = open("1", 'rb')
a = pickle.load(file)
a.to_csv("result.csv", index=False)


