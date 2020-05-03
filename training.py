import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import metrics
from datasets import seq2seq_generator
from networks import bidirectional_ae_2_layer, unidirectional_ae_2_layer
import os
import sys

if len(sys.argv) > 1:
    inp_length = int(sys.argv[1])
else:
    inp_length = 12

# load datasets
train_set = seq2seq_generator('../data/yearly_{}_train.pkl'.format(inp_length + 6))
test_set = seq2seq_generator('../data/yearly_{}_validation.pkl'.format(inp_length + 6))

# define grid search
input_seq_length = hp.HParam('input_seq_length', hp.Discrete([inp_length]))
output_seq_length = hp.HParam('output_seq_length', hp.Discrete([12]))
bottleneck_size = hp.HParam('bottleneck_size', hp.Discrete([25, 50, 100, 200, 250]))
bottleneck_activation = hp.HParam('bottleneck_activation', hp.Discrete(['relu', 'leaky', 'tanh']))
loss_function = hp.HParam('loss_function', hp.Discrete(['mse', 'mae']))
direction = hp.HParam('direction', hp.Discrete(['uni', 'bi']))

# define metrics
mape = metrics.build_mape(overlap=6)
smape = metrics.build_smape(overlap=6)
mase = metrics.build_mase(overlap=6)
owa_estimate = metrics.build_owa(overlap=6)
reconstruction_loss = metrics.build_reconstruction_loss(overlap=6)
metric_names = ['MSE', 'MAE', 'MAPE', 'sMAPE', 'MASE', 'OWA (estimate)', 'Reconstruction Loss']
metrics = ['mse', 'mae', mape, smape, mase, owa_estimate, reconstruction_loss]


# write model training/testing function
def train_test_model(model_generator, hparams, run_name, epochs=10, batch_size=256):
    model = model_generator(hparams, metrics)
    if not os.path.isdir('results/' + str(run_name)):
        os.makedirs('results/' + str(run_name))
    model.fit(train_set, epochs=epochs, steps_per_epoch=len(train_set)//batch_size+1,
              validation_steps=len(test_set)//batch_size+1, validation_data=test_set,
              callbacks=[tf.keras.callbacks.TensorBoard('logs'),
                         hp.KerasCallback('logs', hparams),
                         tf.keras.callbacks.ModelCheckpoint('results/' + str(run_name) + '/best_weights.h5',
                                                            save_best_only=True)])
    return model.evaluate(test_set, steps=84112//batch_size+1)


def run(run_name, model_generator, hparams, epochs=10, batch_size=256):
    with tf.summary.create_file_writer('logs/tuning/' + str(run_name)).as_default():
        hp.hparams(hparams)
        results = train_test_model(model_generator, hparams, run_name, epochs, batch_size)
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

                        for i in range(10):
                            run_name = 'inp_{}__out_{}__loss_{}__bksize_{}__bkact_{}__dir_{}__{}'.format(inp_seq, out_seq,
                                                                                                         loss, bneck_size,
                                                                                                         bneck_activation,
                                                                                                         direct, i)
                            print('-' * 30)
                            print('Starting trial {}: {}'.format(i, run_name))
                            print(hparams)

                            run(run_name, model_generator=model_mapping[direct], hparams=hparams, epochs=5)
