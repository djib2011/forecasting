import tensorflow as tf
import numpy as np
import pickle as pkl
import argparse


def seq2seq_generator(data_path: str, batch_size: int = 256, overlap: int = 6, shuffle: bool = True,
                      augmentation: float = 0, debug: bool = False) -> tf.data.Dataset:
    """
    Factory for building TensorFlow data generators for loading time series data.

    Also supports data augmentation and loading series with overlap for backcast.

    :param data_path: Path of a pickle file that contains two arrays: insample and outsample
    :param batch_size: The batch size
    :param overlap: The length with which x and y will overlap (i.e. if len(insample) == 12 and len(outsample) == 6 and
                    overlap == 5, then len(x) == 12 and len(y) == 11). This is done so that the model is trained for
                    also for backcast.
    :param shuffle: True/False whether or not the data will be shuffled.
    :param augmentation: The percentage of the batch that will be augmented data. E.g. if augmentation == 0.75 and
                         batch_size == 200, then each batch will consist of 50 real series and 150 fake ones.
    :param debug: True/False whether or not to print information about the batches.
    :return: A TensorFlow data generator.
    """
    aug_batch_size = int(batch_size * augmentation)
    real_batch_size = int(batch_size * (1 - augmentation))

    if debug:
        print('---------- Generator ----------')
        print('Augmentation percentage:', augmentation)
        print('Batch size:             ', batch_size)
        print('Real batch size:        ', real_batch_size)
        print('Augmentation batch size:', aug_batch_size)
        print('Max aug num:            ', real_batch_size * (real_batch_size - 1) // 2)
        print('------------------------------')

    def augment(x, y):
        random_ind_1 = tf.random.categorical(tf.math.log([[1.] * real_batch_size]), aug_batch_size)
        random_ind_2 = tf.random.categorical(tf.math.log([[1.] * real_batch_size]), aug_batch_size)

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


def seq2seq_generator_with_aug(data_path, aug_path, batch_size=256, overlap=6, shuffle=True, augmentation=0, debug=False):

    # Load data
    with open(data_path, 'rb') as f:
        x, y = pkl.load(f)

    # Load augmentation data
    aug_size = int(len(x) * augmentation / (1 - augmentation))

    with open(aug_path, 'rb') as f:
        x_aug, y_aug = pkl.load(f)

    if debug:
        print('Augmentation available size:', x_aug.shape[0])

    aug_ind = np.random.permutation(x_aug.shape[0])[:aug_size]
    x_aug = x_aug[aug_ind]
    y_aug = y_aug[aug_ind]

    if debug:
        print('Augmentation target size:', aug_size)
        print('Real size:', x.shape[0])
        print('Synthetic size:', x_aug.shape[0])

    # Combine two sources
    x = np.r_[x, x_aug]
    y = np.r_[y, y_aug]

    if debug:
        print('Final size:', x.shape[0])

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


def seq2seq_generator_decomposed(data_path, batch_size=256, overlap=6, shuffle=True, augmentation=0, debug=False):

    aug_batch_size = int(batch_size * augmentation)
    real_batch_size = int(batch_size * (1 - augmentation))

    if debug:
        print('---------- Generator ----------')
        print('Augmentation percentage:', augmentation)
        print('Batch size:             ', batch_size)
        print('Real batch size:        ', real_batch_size)
        print('Augmentation batch size:', aug_batch_size)
        print('Max aug num:            ', real_batch_size * (real_batch_size - 1) // 2)
        print('-------------------------------')

    def augment(x, y):

        x1 = x[0]
        x2 = x[1]

        random_ind_1 = tf.random.categorical(tf.math.log([[1.] * real_batch_size]), aug_batch_size)
        random_ind_2 = tf.random.categorical(tf.math.log([[1.] * real_batch_size]), aug_batch_size)

        x1_aug = (tf.gather(x1, random_ind_1) + tf.gather(x1, random_ind_2)) / 2
        x2_aug = (tf.gather(x2, random_ind_1) + tf.gather(x2, random_ind_2)) / 2
        y_aug = (tf.gather(y, random_ind_1) + tf.gather(y, random_ind_2)) / 2

        return ((tf.concat([x1, tf.squeeze(x1_aug, [0])], axis=0),
                 tf.concat([x2, tf.squeeze(x2_aug, [0])], axis=0)),
                tf.concat([y, tf.squeeze(y_aug, [0])], axis=0))

        # return ({'x1': tf.concat([x1, tf.squeeze(x1_aug, [0])], axis=0),
        #          'x2': tf.concat([x2, tf.squeeze(x2_aug, [0])], axis=0)},
        #         tf.concat([y, tf.squeeze(y_aug, [0])], axis=0))

    # Load data
    with open(data_path, 'rb') as f:
        x1, x2, y = pkl.load(f)

    # Overlap input with output
    if overlap:
        y = np.c_[(x1 + x2)[:, -overlap:], y]

    x1 = x1[..., np.newaxis]
    x2 = x2[..., np.newaxis]
    y = y[..., np.newaxis]

    # Tensorflow dataset
    data = tf.data.Dataset.from_tensor_slices(((x1, x2), y))
    if shuffle:
        data = data.shuffle(buffer_size=len(y))
    data = data.repeat()
    data = data.batch(batch_size=real_batch_size)
    if augmentation:
        data = data.map(augment)
    data = data.prefetch(buffer_size=1)

    data.__class__ = type(data.__class__.__name__, (data.__class__,), {'__len__': lambda self: len(y)})
    return data


def seq2seq_generator_only_aug(data_path: str, batch_size: int = 256, overlap: int = 6, shuffle: bool = True) -> tf.data.Dataset:

    def augment(x, y):
        random_ind_1 = tf.random.categorical(tf.math.log([[1.] * batch_size]), batch_size)
        random_ind_2 = tf.random.categorical(tf.math.log([[1.] * batch_size]), batch_size)
        random_ind_3 = tf.random.categorical(tf.math.log([[1.] * batch_size]), batch_size)
        random_ind_4 = tf.random.categorical(tf.math.log([[1.] * batch_size]), batch_size)
        random_ind_5 = tf.random.categorical(tf.math.log([[1.] * batch_size]), batch_size)

        x_aug = (tf.gather(x, random_ind_1) + tf.gather(x, random_ind_2) + tf.gather(x, random_ind_3)
                 + tf.gather(x, random_ind_4) + tf.gather(x, random_ind_5)) / 2

        y_aug = (tf.gather(y, random_ind_1) + tf.gather(y, random_ind_2) + tf.gather(y, random_ind_3)
                 + tf.gather(y, random_ind_4) + tf.gather(y, random_ind_5)) / 2

        return tf.squeeze(x_aug, [0]), tf.squeeze(y_aug, [0])

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
    data = data.map(augment)
    data = data.prefetch(buffer_size=1)

    data.__class__ = type(data.__class__.__name__, (data.__class__,), {'__len__': lambda self: len(x)})
    return data



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_len', type=int, default=12, help='Insample length.')
    parser.add_argument('-o', '--overlap', type=int, default=6, help='Length of overlap between input and output. '
                                                                     'Outsample length is overlap + 6.')
    parser.add_argument('-a', '--aug', type=float, default=0., help='Percentage of augmented series in batch')
    parser.add_argument('-d', '--decomposed', action='store_true', help='Deompose inputs.')
    parser.add_argument('--line', action='store_true', help='Approximate outsample with a linear regression.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode: Print lots of diagnostic messages.')

    args = parser.parse_args()

    inp_length = args.input_len
    overlap = args.overlap
    out_length = overlap + 6

    if args.line:
        train_set = 'data/yearly_{}_train_line.pkl'.format(inp_length + 6)
        test_set = 'data/yearly_{}_validation_line.pkl'.format(inp_length + 6)
        gen = seq2seq_generator
    elif args.decomposed:
        train_set = 'data/yearly_{}_train_decomposed.pkl'.format(inp_length + 6)
        test_set = 'data/yearly_{}_validation_decomposed.pkl'.format(inp_length + 6)
        gen = seq2seq_generator_decomposed
    else:
        train_set = 'data/yearly_{}_train.pkl'.format(inp_length + 6)
        test_set = 'data/yearly_{}_validation.pkl'.format(inp_length + 6)
        gen = seq2seq_generator

    if args.decomposed:
        train_gen = seq2seq_generator_decomposed(train_set, batch_size=256, overlap=overlap, shuffle=True,
                                                 augmentation=args.aug, debug=args.debug)
        test_gen = seq2seq_generator_decomposed(test_set, batch_size=256, overlap=overlap, shuffle=True,
                                                augmentation=0, debug=args.debug)

        for (x1, x2), y in train_gen:
            print('Train set:')
            print(x1.shape, x2.shape, y.shape)
            break

        for (x1, x2), y in test_gen:
            print('Test set:')
            print(x1.shape, x2.shape, y.shape)
            break

    else:
        train_gen = seq2seq_generator(train_set, batch_size=256, overlap=overlap, shuffle=True,
                                      augmentation=args.aug, debug=args.debug)
        test_gen = seq2seq_generator(test_set, batch_size=256, overlap=overlap, shuffle=True,
                                     augmentation=0, debug=args.debug)

        for x, y in train_gen:
            print('Train set:')
            print(x.shape, y.shape)
            break

        for x, y in test_gen:
            print('Test set:')
            print(x.shape, y.shape)
            break
