import tensorflow as tf
import utils


def simple_attention(hparams, metrics):

    if 'conv' in hparams['direction']:
        layer_type = 'conv'
    elif 'uni' in hparams['direction']:
        layer_type = 'lstm'
    elif 'bi' in hparams['direction']:
        layer_type = 'bi'
    else:
        raise ValueError('Unsupported model type')

    num_layers = int([x for x in hparams['direction'] if x.isdigit()][0])

    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    eb = utils.layers.EncoderBlock(layer_type=layer_type, num_layers=num_layers,
                                   kernel_size=hparams['kernel_size'])(inp)
    ab = utils.layers.AttentionBottleneck(hparams['output_seq_length'], hparams['bottleneck_size'],
                                          layer_type=layer_type, kernel_size=hparams['kernel_size'],
                                          bottleneck_activation=hparams['bottleneck_activation'])(eb)
    db = utils.layers.DecoderBlock(layer_type=layer_type, num_layers=num_layers, kernel_size=hparams['kernel_size'])(ab)
    out = tf.keras.layers.Dense(1)(db)

    model = tf.keras.models.Model(inp, out)

    if hparams['optimizer'] == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'])
    elif hparams['optimizer'] == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=hparams['learning_rate'])
    elif hparams['optimizer'] == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'])
    else:
        raise ValueError('Invalid value for "optimizer".')

    model.compile(loss=hparams['loss_function'], optimizer=opt, metrics=metrics)

    return model
