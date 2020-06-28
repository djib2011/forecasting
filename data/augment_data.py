import pickle as pkl
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=12, help='Insample length.')
parser.add_argument('-n', '--num_samples', type=int, default=12, help='Number of samples after augmentation.')

args = parser.parse_args()

window = args.input_len + 6

# Load data
with open('yearly_{}_train.pkl'.format(window), 'rb') as f:
    data = pkl.load(f)

# Generate synthetic samples
s1 = np.random.choice(np.arange(len(data[0])), args.num_samples, replace=True)
s2 = np.random.choice(np.arange(len(data[0])), args.num_samples, replace=True)

syn = [(d[s1] + d[s2]) / 2 for d in data]

# Store new
with open('yearly_{}_train_aug.pkl'.format(window), 'wb') as f:
    pkl.dump(syn, f)
