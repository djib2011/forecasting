from tqdm import tqdm
import pickle as pkl
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression


def decompose(points):
    lr = LinearRegression()
    x = np.arange(len(points)).reshape(-1, 1)
    lr.fit(x, points)
    line = lr.predict(x)
    return line, points - line


def run_decomposition(dataset):
    # Decompose time series
    trends, residuals = [], []

    for ts in tqdm(dataset[0]):
        trend, residual = decompose(ts)
        trends.append(trend)
        residuals.append(residual)

    return trends, residuals, dataset[1]


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_len', type=int, default=12, help='Insample length.')

    args = parser.parse_args()

    window = args.input_len + 6

    # Load data
    with open('yearly_{}_train.pkl'.format(window), 'rb') as f:
        train = pkl.load(f)
    with open('yearly_{}_validation.pkl'.format(window), 'rb') as f:
        test = pkl.load(f)

    # Run the decomposition
    trends_train, residuals_train, y_train = run_decomposition(train)
    trends_test, residuals_test, y_test = run_decomposition(test)

    # Store data
    with open('yearly_{}_train_decomposed.pkl'.format(window), 'wb') as f:
        pkl.dump((np.array(trends_train), np.array(residuals_train), y_train), f)
    with open('yearly_{}_validation_decomposed.pkl'.format(window), 'wb') as f:
        pkl.dump((np.array(trends_test), np.array(residuals_test), y_test), f)
