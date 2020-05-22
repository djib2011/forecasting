import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import metrics
from datasets import seq2seq_generator
from networks import bidirectional_ae_2_layer, unidirectional_ae_2_layer
import os
import sys
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=12, help='Insample length.')
parser.add_argument('-o', '--overlap', type=int, default=6, help='Length of overlap between input and output. '
                                                                  'Outsample length is overlap + 6.')
parser.add_argument('--line', action='store_true', help='Approximate outsample with a linear regression.')
parser.add_argument('--no_logs', action='store_false', help='Don\'t store log files for any of the runs')
parser.add_argument('--debug', action='store_true', help='Don\'t train any of the models.')

args = parser.parse_args()

inp_length = args.input_len
overlap = args.overlap
out_length = overlap + 6

# load datasets
if args.line:
    train_set = seq2seq_generator('data/yearly_{}_train_line.pkl'.format(inp_length + 6), overlap=overlap)
    test_set = seq2seq_generator('data/yearly_{}_validation_line.pkl'.format(inp_length + 6), overlap=overlap)
else:
    train_set = seq2seq_generator('data/yearly_{}_train.pkl'.format(inp_length + 6), overlap=overlap)
    test_set = seq2seq_generator('data/yearly_{}_validation.pkl'.format(inp_length + 6), overlap=overlap)

# define grid search
input_seq_length = hp.HParam('input_seq_length', hp.Discrete([inp_length]))
output_seq_length = hp.HParam('output_seq_length', hp.Discrete([out_length]))
bottleneck_size = hp.HParam('bottleneck_size', hp.Discrete([200, 250]))
bottleneck_activation = hp.HParam('bottleneck_activation', hp.Discrete(['relu', 'leaky', 'tanh']))
loss_function = hp.HParam('loss_function', hp.Discrete(['mae']))
direction = hp.HParam('direction', hp.Discrete(['bi']))

# define metrics
reconstruction_loss = metrics.build_reconstruction_loss(overlap=overlap)
mape = metrics.build_mape(overlap=overlap)
smape = metrics.build_smape(overlap=overlap)
metric_names = ['MSE', 'MAE', 'MAPE', 'sMAPE', 'Reconstruction Loss', ]
metric_functions = ['mse', 'mae', mape, smape, reconstruction_loss]

if overlap:
    # MASE, OWA can't be estimated without overlap
    mase = metrics.build_mase(overlap=overlap)
    owa_estimate = metrics.build_owa(overlap=overlap)
    metric_names.extend(['MASE', 'OWA (estimate)'])
    metric_functions.extend([mase, owa_estimate])

# write model training/testing function
def train_test_model(model_generator, hparams, run_name, epochs=10, batch_size=256, logs=True):
    model = model_generator(hparams, metric_functions)
    if not os.path.isdir('results/' + str(run_name)):
        os.makedirs('results/' + str(run_name))

    callbacks = [tf.keras.callbacks.ModelCheckpoint('results/' + str(run_name) + '/best_weights.h5',
                                                    save_best_only=True)]

    if logs:
        callbacks.extend([tf.keras.callbacks.TensorBoard('logs'), hp.KerasCallback('logs', hparams)])

    model.fit(train_set, epochs=epochs, steps_per_epoch=len(train_set)//batch_size+1,
              validation_steps=len(test_set)//batch_size+1, validation_data=test_set,
              callbacks=callbacks)
    return model.evaluate(test_set, steps=len(test_set)//batch_size+1)


def run(run_name, model_generator, hparams, epochs=10, batch_size=256, logs=True):
    with tf.summary.create_file_writer('logs/tuning/' + str(run_name)).as_default():
        hp.hparams(hparams)
        results = train_test_model(model_generator, hparams, run_name, epochs, batch_size, logs=logs)
        for name, value in zip(metric_names, results[1:]):
            tf.summary.scalar(name, value, step=epochs)


model_mapping = {'uni': unidirectional_ae_2_layer, 'bi': bidirectional_ae_2_layer}

with tf.summary.create_file_writer('logs/tuning').as_default():
    hp.hparams_config(
        hparams=[input_seq_length, output_seq_length, bottleneck_size, bottleneck_activation,
                 loss_function, direction],
        metrics=[hp.Metric(name, display_name=name) for name in metric_names]
    )


# Start training loop
def hyperparam_loop(logs=True, line=False):
    for inp_seq in input_seq_length.domain.values:
        for out_seq in output_seq_length.domain.values:
            for loss in loss_function.domain.values:
                for bneck_size in bottleneck_size.domain.values:
                    for bneck_activation in bottleneck_activation.domain.values:
                        for direct in direction.domain.values:
                            hparams = {'bottleneck_size': bneck_size,
                                       'bottleneck_activation': bneck_activation,
                                       'direction': direct,
                                       'input_seq_length': inp_seq,
                                       'output_seq_length': out_seq,
                                       'loss_function': loss}

                            for i in range(30):
                                run_name = 'inp_{}__out_{}__loss_{}__bksize_{}__bkact_{}__dir_{}__{}'.format(inp_seq, out_seq,
                                                                                                             loss, bneck_size,
                                                                                                             bneck_activation,
                                                                                                             direct, i)
                                if line:
                                    run_name = 'line__' + run_name

                                print('-' * 30)
                                print('Starting trial {}: {}'.format(i, run_name))
                                print(hparams)
                                if args.debug:
                                    continue
                                run(run_name, model_generator=model_mapping[direct],
                                    hparams=hparams, epochs=5, logs=logs)


if __name__ == '__main__':
    hyperparam_loop(logs=(not args.no_logs), line=args.line)
