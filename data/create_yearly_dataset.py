from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle as pkl
import tensorflow as tf
import sys
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('once')

data_path = Path('../../M4-methods/Dataset/Train/Yearly-train.csv')

data = pd.read_csv(data_path)
data = data.drop('V1', axis=1)

inp_len = 12
overlap = 6

if len(sys.argv) > 1:
    inp_len = int(sys.argv[1])

if len(sys.argv) > 2:
    overlap = int(sys.argv[2])

window = inp_len + 6


def split_series(ser):
    x = []
    y = []
    for i in range(ser.notna().sum() - window +1):
        x.append(ser[i:i+window-overlap])
        y.append(ser[i+window-overlap:i+window])
    return np.array(x), np.array(y)


for s in tqdm(data.values):
    ser = pd.Series(s)
    if ser.notna().sum() <= window-1:
        continue
    x, y = split_series(ser)
    try:
        X = np.vstack([X, x])
        Y = np.vstack([Y, y])
    except NameError:
        X = x.copy()
        Y = y.copy()
        
        
X_train, X_test, y_train, y_test = train_test_split(X, Y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

sc_train = MinMaxScaler()
X_train = sc_train.fit_transform(X_train.T).T
y_train = sc_train.transform(y_train.T).T

sc_test = MinMaxScaler()
X_test = sc_train.fit_transform(X_test.T).T
y_test = sc_train.transform(y_test.T).T

pkl.dump(sc_train, open('yearly_{}_{}_scales_train.pkl'.format(window, overlap), 'wb'))
pkl.dump(sc_test, open('yearly_{}_{}_scales_test.pkl'.format(window, overlap), 'wb'))
pkl.dump((X_train, y_train), open('yearly_{}_{}_train.pkl'.format(window, overlap), 'wb'))
pkl.dump((X_test, y_test), open('yearly_{}_{}_validation.pkl'.format(window, overlap), 'wb'))

print('Saved files:')
print('yearly_{}_{}_scales_train.pkl'.format(window, overlap))
print('yearly_{}_{}_scales_test.pkl'.format(window, overlap))
print('yearly_{}_{}_train.pkl'.format(window, overlap))
print('yearly_{}_{}_validation.pkl'.format(window, overlap))

