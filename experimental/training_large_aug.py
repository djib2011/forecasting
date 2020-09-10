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
parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')

args = parser.parse_args()

inp_length = args.input_len
overlap = args.overlap
out_length = overlap + 6

batch_size = 256 * 2 * 2 * 2 * 2 * 2

# load datasets
train_set = datasets.seq2seq_generator_only_aug('data/yearly_{}_train.pkl'.format(inp_length + 6), overlap=overlap,
                                                batch_size=batch_size)
test_set = datasets.seq2seq_generator('data/yearly_{}_validation.pkl'.format(inp_length + 6), overlap=overlap,
                                      batch_size=batch_size, augmentation=0)

# define grid search
input_seq_length = hp.HParam('input_seq_length', hp.Discrete([inp_length]))
output_seq_length = hp.HParam('output_seq_length', hp.Discrete([out_length]))
augmentation = hp.HParam('augmentation', hp.Discrete([1.0]))
bottleneck_size = hp.HParam('bottleneck_size', hp.Discrete([700]))
bottleneck_activation = hp.HParam('bottleneck_activation', hp.Discrete(['relu']))
loss_function = hp.HParam('loss_function', hp.Discrete(['mae']))
direction = hp.HParam('direction', hp.Discrete(['conv4']))
kernel_size = hp.HParam('kernel_size', hp.Discrete([5]))
optimizer = hp.HParam('optimizer', hp.Discrete(['adam']))
learning_rate = hp.HParam('learning_rate', hp.Discrete([0.001]))

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
def train_test_model(model_generator, hparams, run_name, cycles=10, batch_size=256, num_run=0):
    model = model_generator(hparams, metric_functions)

    epochs = cycles + 5
    result_dir = 'results/large_aug/'

    callbacks = [utils.callbacks.SnapshotWithAveraging(result_dir + str(run_name), n_cycles=cycles,
                                                           max_epochs=epochs, steps_to_average=100,
                                                           min_warmup_epochs=1, cold_start_id=num_run)]

    model.fit(train_set, epochs=epochs, steps_per_epoch=len(train_set)//batch_size+1,
              validation_steps=len(test_set)//batch_size+1, validation_data=test_set,
              callbacks=callbacks)


def run(run_name, model_generator, hparams, cycles=10, batch_size=512, num_run=0):
    train_test_model(model_generator, hparams, run_name, cycles, batch_size, num_run=num_run)


model_mapping = {'uni': networks.unidirectional_ae_2_layer, 'fc': networks.fully_connected_ae_2_layer,
                 'bi': networks.bidirectional_ae_2_layer, 'bi2': networks.bidirectional_ae_3_layer,
                 'conv': networks.convolutional_ae_2_layer, 'conv2': networks.convolutional_ae_3_layer,
                 'conv3': networks.convolutional_ae_4_layer, 'conv4': networks.convolutional_ae_5_layer,
                 'comb': networks.combined_ae_2_layer, 'cat': networks.concat_ae_2_layer,
                 'dual_inp': networks.convolutional_4_layer_2_input}

attn_names = ['attn_conv' + str(i) for i in range(1, 6)] + \
             ['attn_uni' + str(i) for i in range(1, 6)] + \
             ['attn_bi' + str(i) for i in range(1, 6)]

model_mapping.update({a: networks.simple_attention for a in attn_names})

# Start training loop
def hyperparam_loop_new(cycles=15, cold_restarts=4, batch_size=256):

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
                                                    hparams=hparams, cycles=cycles, batch_size=batch_size, num_run=num_run)


if __name__ == '__main__':

    hyperparam_loop_new(cycles=1000, cold_restarts=10, batch_size=batch_size)
