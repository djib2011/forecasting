import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import metrics
import datasets
import networks
import os
import argparse

import utils

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=12, help='Insample length.')
parser.add_argument('-o', '--overlap', type=int, default=6, help='Length of overlap between input and output. '
                                                                  'Outsample length is overlap + 6.')
parser.add_argument('-a', '--aug', type=float, default=0., help='Percentage of augmented series in batch')
parser.add_argument('-d', '--decomposed', action='store_true', help='Deompose inputs.')
parser.add_argument('--line', action='store_true', help='Approximate outsample with a linear regression.')
parser.add_argument('--no_logs', action='store_false', help='Don\'t store log files for any of the runs')
parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')
parser.add_argument('--no_snapshot', action='store_true', help='Don\'t use snapshot ensembles; instead make full runs')

args = parser.parse_args()

inp_length = args.input_len
overlap = args.overlap
out_length = overlap + 6

batch_size = 256
if args.aug >= 0.5:
    batch_size *= 2
if args.aug >= 0.75:
    batch_size *= 2
if args.aug >= 0.875:
    batch_size *= 2
if args.aug >= 0.9375:
    batch_size *= 2
if args.aug >= 0.96875:
   batch_size *= 2
# if args.aug >= 0.984375:
#     batch_size *= 2

# load datasets
if args.line:
    train_set = datasets.seq2seq_generator('data/yearly_{}_train_line.pkl'.format(inp_length + 6), overlap=overlap,
                                            batch_size=batch_size, augmentation=args.aug, debug=args.debug)
    test_set = datasets.seq2seq_generator('data/yearly_{}_validation_line.pkl'.format(inp_length + 6), overlap=overlap,
                                          batch_size=batch_size, augmentation=0)
elif args.decomposed:
    train_set = datasets.seq2seq_generator_decomposed('data/yearly_{}_train_decomposed.pkl'.format(inp_length + 6), overlap=overlap,
                                                      batch_size=batch_size, augmentation=args.aug, debug=args.debug)
    test_set = datasets.seq2seq_generator_decomposed('data/yearly_{}_validation_decomposed.pkl'.format(inp_length + 6), overlap=overlap,
                                                      batch_size=batch_size, augmentation=0)
else:
    train_set = datasets.seq2seq_generator('data/yearly_{}_train.pkl'.format(inp_length + 6), overlap=overlap,
                                           batch_size=batch_size, augmentation=args.aug, debug=args.debug)
    test_set = datasets.seq2seq_generator('data/yearly_{}_validation.pkl'.format(inp_length + 6), overlap=overlap,
                                          batch_size=batch_size, augmentation=0)

# define grid search
input_seq_length = hp.HParam('input_seq_length', hp.Discrete([inp_length]))
output_seq_length = hp.HParam('output_seq_length', hp.Discrete([out_length]))
augmentation = hp.HParam('direction', hp.Discrete([args.aug]))
bottleneck_size = hp.HParam('bottleneck_size', hp.Discrete([700]))
bottleneck_activation = hp.HParam('bottleneck_activation', hp.Discrete(['relu', 'gelu']))
loss_function = hp.HParam('loss_function', hp.Discrete(['mae']))
direction = hp.HParam('direction', hp.Discrete(['conv3']))
kernel_size = hp.HParam('kernel_size', hp.Discrete([2, 3, 4, 5, 6]))
optimizer = hp.HParam('optimizer', hp.Discrete(['adam']))
learning_rate = hp.HParam('learning_rate', hp.Discrete([0.01, 0.005, 0.001, 0.0005, 0.0001]))

# define metrics
reconstruction_loss = metrics.build_reconstruction_loss(overlap=overlap)
mape = metrics.build_mape(overlap=overlap)
smape = metrics.build_smape(overlap=overlap)
metric_names = ['MSE', 'MAE', 'MAPE', 'sMAPE', 'Reconstruction Loss']
metric_functions = ['mse', 'mae', mape, smape, reconstruction_loss]

if overlap:
    # MASE, OWA can't be estimated without overlap
    mase = metrics.build_mase(overlap=overlap)
    owa_estimate = metrics.build_owa(overlap=overlap)
    metric_names.extend(['MASE', 'OWA (estimate)'])
    metric_functions.extend([mase, owa_estimate])


# write model training/testing function
def train_test_model(model_generator, hparams, run_name, epochs=10, batch_size=256, logs=True, num_run=0):
    model = model_generator(hparams, metric_functions)
    cycles = 15

    if args.no_snapshot:
        if not os.path.isdir('results/' + str(run_name)):
            os.makedirs('results/' + str(run_name))
        callbacks = [tf.keras.callbacks.ModelCheckpoint('results/' + str(run_name) + '/best_weights.h5',
                                                        save_best_only=True)]
    else:
        epochs = cycles + 5
        callbacks = [utils.callbacks.SnapshotWithAveraging('results/' + str(run_name), n_cycles=cycles,
                                                           max_epochs=epochs, steps_to_average=100,
                                                           cold_start_id=num_run)]

    if logs:
        callbacks.extend([tf.keras.callbacks.TensorBoard('logs'), hp.KerasCallback('logs', hparams)])

    model.fit(train_set, epochs=epochs, steps_per_epoch=len(train_set)//batch_size+1,
              validation_steps=len(test_set)//batch_size+1, validation_data=test_set,
              callbacks=callbacks)
    return model.evaluate(test_set, steps=len(test_set)//batch_size+1)


def run(run_name, model_generator, hparams, epochs=10, batch_size=512, logs=True, num_run=0):
    with tf.summary.create_file_writer('logs/tuning/' + str(run_name)).as_default():
        hp.hparams(hparams)
        results = train_test_model(model_generator, hparams, run_name, epochs, batch_size, logs=logs, num_run=num_run)
        for name, value in zip(metric_names, results[1:]):
            tf.summary.scalar(name, value, step=epochs)


model_mapping = {'uni': networks.unidirectional_ae_2_layer, 'fc': networks.fully_connected_ae_2_layer,
                 'bi': networks.bidirectional_ae_2_layer, 'bi2': networks.bidirectional_ae_3_layer,
                 'conv': networks.convolutional_ae_2_layer, 'conv2': networks.convolutional_ae_3_layer,
                 'conv3': networks.convolutional_ae_4_layer, 'conv4': networks.convolutional_ae_5_layer,
                 'comb': networks.combined_ae_2_layer, 'cat': networks.concat_ae_2_layer,
                 'dual_inp': networks.convolutional_4_layer_2_input}

with tf.summary.create_file_writer('logs/tuning').as_default():
    h = [input_seq_length, output_seq_length, bottleneck_size, bottleneck_activation, loss_function]
    if args.decomposed:
        h.append(direction)
    hp.hparams_config(
        hparams=h,
        metrics=[hp.Metric(name, display_name=name) for name in metric_names]
    )


# Start training loop
def hyperparam_loop_old(logs=True, line=False, epochs=5, batch_size=256):
    if args.debug:
        print('Running generic optimization loop.')
    for inp_seq in input_seq_length.domain.values:
        for out_seq in output_seq_length.domain.values:
            for aug in augmentation.domain.values:
                for loss in loss_function.domain.values:
                    for bneck_size in bottleneck_size.domain.values:
                        for bneck_activation in bottleneck_activation.domain.values:
                            for direct in direction.domain.values:
                                hparams = {'bottleneck_size': bneck_size,
                                           'bottleneck_activation': bneck_activation,
                                           'direction': direct,
                                           'input_seq_length': inp_seq,
                                           'output_seq_length': out_seq,
                                           'loss_function': loss,
                                           'kernel_size': (3,),
                                           'stride': 1}

                                if args.no_snapshot:
                                    for i in range(30):
                                        run_name = 'inp_{}__out_{}__aug_{}__loss_{}__bksize_{}__bkact_{}__dir_{}__{}'.format(inp_seq, out_seq,
                                                                                                                             aug, loss, bneck_size,
                                                                                                                             bneck_activation, direct, i)
                                        if line:
                                            run_name = 'line__' + run_name

                                        print('-' * 30)
                                        print('Starting trial {}: {}'.format(i, run_name))
                                        print(hparams)
                                        if args.debug:
                                            model = model_mapping[direct](hparams, metric_functions)
                                            if args.decomposed:
                                                x, y = next(iter(train_set))
                                                print('Batch shape', x[0].shape, x[1].shape, y.shape)
                                                model.train_on_batch(x, y)
                                            else:
                                                x, y = next(iter(train_set))
                                                print('Batch shape', x.shape, y.shape)
                                                model.train_on_batch(x, y)
                                            continue
                                        run(run_name, model_generator=model_mapping[direct], epochs=epochs,
                                            hparams=hparams, logs=logs, batch_size=batch_size)

                                else:
                                    raise NotImplementedError('Snapshots not available for old hyperparameter loop')


def hyperparam_loop_new(cycles=15, cold_restarts=4, batch_size=256):

    if args.debug:
        print('Running CNN-specialized optimization loop.')
    for inp_seq in input_seq_length.domain.values:
        for out_seq in output_seq_length.domain.values:
            for aug in augmentation.domain.values:
                for loss in loss_function.domain.values:
                    for bneck_size in bottleneck_size.domain.values:
                        for bneck_activation in bottleneck_activation.domain.values:
                            for direct in direction.domain.values:
                                for ksize in kernel_size.domain.values:
                                    for opt in optimizer.domain.values:
                                        for lr in learning_rate.domain.values:
                                            hparams = {'bottleneck_size': bneck_size,
                                                       'bottleneck_activation': bneck_activation,
                                                       'direction': direct,
                                                       'input_seq_length': inp_seq,
                                                       'output_seq_length': out_seq,
                                                       'loss_function': loss,
                                                       'kernel_size': ksize,
                                                       'stride': 1,
                                                       'optimizer': opt,
                                                       'learning_rate': lr}

                                            if args.no_snapshot:
                                                raise NotImplementedError('New hyperparameter loop only available with'
                                                                          'snapshot ensembling.')
                                            else:
                                                family_name = 'inp_{}__out_{}__aug_{}__loss_{}__bksize_{}__bkact_{}__' \
                                                              'dir_{}__ksize_{}__opt_{}__lr_{}'.format(inp_seq, out_seq,
                                                                                                       aug, loss, bneck_size,
                                                                                                       bneck_activation,
                                                                                                       direct, ksize, opt,
                                                                                                       lr)
                                                for num_run in range(cold_restarts):
                                                    print('-' * 30)
                                                    print('Starting trial: {}  (run {})'.format(family_name, num_run))
                                                    print(hparams)
                                                    if args.debug:
                                                        model = model_mapping[direct](hparams, metric_functions)
                                                        x, y = next(iter(train_set))
                                                        print('Batch shape', x.shape, y.shape)
                                                        model.train_on_batch(x, y)
                                                        continue

                                                    run(family_name, model_generator=model_mapping[direct],
                                                        hparams=hparams, epochs=cycles, batch_size=batch_size, num_run=num_run)


if __name__ == '__main__':
    max_epochs = 20
    epochs = min(max_epochs, int(5 / (1 - args.aug)))
    real_batch_size = int(batch_size * (1 - args.aug))

    # hyperparam_loop_new(logs=(not args.no_logs), line=args.line, epochs=1, batch_size=real_batch_size)
    hyperparam_loop_new(cycles=15, cold_restarts=4, batch_size=real_batch_size)
