import tensorflow as tf
import numpy as np
import pickle as pkl


def seq2seq_generator(data_path, batch_size=256, overlap=6, shuffle=True, augmentation=0):

    aug_batch_size = int(batch_size * augmentation)
    real_batch_size = int(batch_size * (1 - augmentation))

    def augment(x, y):
        random_ind_1 = tf.random.categorical(tf.math.log([[1.] * aug_batch_size]), aug_batch_size)
        random_ind_2 = tf.random.categorical(tf.math.log([[1.] * aug_batch_size]), aug_batch_size)

        x_aug = (tf.gather(x, random_ind_1) + tf.gather(x, random_ind_2)) / 2
        y_aug = (tf.gather(y, random_ind_1) + tf.gather(y, random_ind_2)) / 2

        return tf.concat([x, tf.squeeze(x_aug, [0])], axis=0), tf.concat([y, tf.squeeze(y_aug, [0])], axis=0)

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
    data = data.batch(batch_size=real_batch_size)
    if augmentation:
        data = data.map(augment)
    data = data.prefetch(buffer_size=1)

    data.__class__ = type(data.__class__.__name__, (data.__class__,), {'__len__': lambda self: len(x)})
    return data
