import pickle

'''
load data from "1" file.
'''
file = open("1", 'rb')
a, b = pickle.load(file)
print(a.head())

print("second")
print(b.head())