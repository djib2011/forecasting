import tensorflow as tf
import numpy as np
import pickle as pkl


def seq2seq_generator(data_path, batch_size=256, overlap=6, shuffle=True):

    # Load data
    with open(data_path, 'rb') as f:
        x, y = pkl.load(f)

    # Overlap input with output
    if overlap:
        y = np.c_[x[:, -overlap:], y]

    x = x[..., np.newaxis]
    y = y[..., np.newaxis]

    # Tensorflow dataset
    data = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        data = data.shuffle(buffer_size=len(x))
    data = data.repeat()
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=1)

    data.__class__ = type(data.__class__.__name__, (data.__class__,), {'__len__': lambda self: len(x)})
    return data
