from networks.convolutional import *
from networks.sequential import *
from networks.misc import *
from networks.attention import *


if __name__ == '__main__':

    hparams1 = {'bottleneck_size': 700,
                'bottleneck_activation': 'gelu',
                'direction': 'conv4',
                'input_seq_length': 18,
                'output_seq_length': 14,
                'loss_function': 'mae',
                'kernel_size': 3,
                'optimizer': 'adam',
                'stride': 1,
                'learning_rate': 0.01}

    metrics = []

    model = convolutional_ae_4_layer(hparams1, metrics)
    model.summary()

    x, y = tf.zeros((100, 18, 1)), tf.zeros((100, 14, 1))

    model_types = ['attn_conv2', 'attn_uni3', 'attn_bi4', 'attn_conv4']
    for mt in model_types:
        hparams = {'bottleneck_size': 700,
                   'bottleneck_activation': 'relu',
                   'direction': mt,
                   'input_seq_length': 18,
                   'output_seq_length': 14,
                   'loss_function': 'mae',
                   'kernel_size': 3,
                   'optimizer': 'adam',
                   'stride': 1,
                   'learning_rate': 0.01}

        model = simple_attention(hparams, metrics)

        model.fit(x, y, verbose=0)

    model.summary()
