import pickle
import pandas

'''
load data from "1" file.
'''
file = open("1", 'rb')
a = pickle.load(file)
a.to_csv("result.csv")
