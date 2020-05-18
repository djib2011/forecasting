from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler
import argparse
import warnings
import os

warnings.filterwarnings('once')

if not os.getcwd().endswith('data'):
    os.chdir('data')

data_path = Path('Yearly-train.csv')

data = pd.read_csv(data_path)
data = data.drop('V1', axis=1)

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=12, help='Insample length.')
parser.add_argument('--line', action='store_true', help='Approximate outsample with a linear regression.')

args = parser.parse_args()

window = args.input_len + 6

if args.line:
    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()

    def best_line(y):
        lin_reg.fit(np.arange(len(y))[:, np.newaxis], y[:, np.newaxis])
        return lin_reg.predict(np.arange(len(y))[:, np.newaxis]).flatten()


def split_series(ser):
    x = []
    y = []
    for i in range(ser.notna().sum() - window + 1):
        x.append(ser[i:i + window - 6])
        if args.line:
            y.append(best_line(ser[i + window - 6:i + window]))
        else:
            y.append(ser[i + window - 6:i + window])
    return np.array(x), np.array(y)


for s in tqdm(data.values):
    ser = pd.Series(s)
    if ser.notna().sum() <= window - 1:
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

if args.line:
    pkl.dump(sc_train, open('yearly_{}_scales_train_line.pkl'.format(window), 'wb'))
    pkl.dump(sc_test, open('yearly_{}_scales_test_line.pkl'.format(window), 'wb'))
    pkl.dump((X_train, y_train), open('yearly_{}_train_line.pkl'.format(window), 'wb'))
    pkl.dump((X_test, y_test), open('yearly_{}_validation_line.pkl'.format(window), 'wb'))

    print('Saved files:')
    print('yearly_{}_scales_train_line.pkl'.format(window))
    print('yearly_{}_scales_test_line.pkl'.format(window))
    print('yearly_{}_train_line.pkl'.format(window))
    print('yearly_{}_validation_line.pkl'.format(window))

else:
    pkl.dump(sc_train, open('yearly_{}_scales_train.pkl'.format(window), 'wb'))
    pkl.dump(sc_test, open('yearly_{}_scales_test.pkl'.format(window), 'wb'))
    pkl.dump((X_train, y_train), open('yearly_{}_train.pkl'.format(window), 'wb'))
    pkl.dump((X_test, y_test), open('yearly_{}_validation.pkl'.format(window), 'wb'))

    print('Saved files:')
    print('yearly_{}_scales_train.pkl'.format(window))
    print('yearly_{}_scales_test.pkl'.format(window))
    print('yearly_{}_train.pkl'.format(window))
    print('yearly_{}_validation.pkl'.format(window))
