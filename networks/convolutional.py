import tensorflow as tf
import tensorflow_addons as tfa


def convolutional_ae_2_layer(hparams, metrics):
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Conv1D(64, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same')(inp)
    x = tf.keras.layers.Conv1D(16, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['bottleneck_size'])(x)
    if hparams['bottleneck_activation'] == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif hparams['bottleneck_activation'] == 'leaky':
        x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(16 * 2 * hparams['output_seq_length'])(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], 16 * 2))(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same')(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same')(x)
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inp, out)

    model.compile(loss=hparams['loss_function'], optimizer='adam', metrics=metrics)

    return model


def convolutional_ae_3_layer(hparams, metrics):
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Conv1D(128, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same')(inp)
    x = tf.keras.layers.Conv1D(64, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same')(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['bottleneck_size'])(x)
    if hparams['bottleneck_activation'] == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif hparams['bottleneck_activation'] == 'leaky':
        x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(16 * 2 * hparams['output_seq_length'])(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], 16 * 2))(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same')(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same')(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same')(x)
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inp, out)

    model.compile(loss=hparams['loss_function'], optimizer='adam', metrics=metrics)

    return model


def convolutional_ae_4_layer(hparams, metrics):
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Conv1D(256, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(inp)
    x = tf.keras.layers.Conv1D(128, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['bottleneck_size'])(x)
    if hparams['bottleneck_activation'] == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif hparams['bottleneck_activation'] == 'leaky':
        x = tf.keras.layers.LeakyReLU()(x)
    elif hparams['bottleneck_activation'] == 'gelu':
        x = tfa.layers.GELU()(x)
    x = tf.keras.layers.Dense(16 * 2 * hparams['output_seq_length'])(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], 16 * 2))(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv1D(256, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    out = tf.keras.layers.Dense(1)(x)

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


def convolutional_ae_5_layer(hparams, metrics):
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Conv1D(512, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(inp)
    x = tf.keras.layers.Conv1D(256, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['bottleneck_size'])(x)
    if hparams['bottleneck_activation'] == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif hparams['bottleneck_activation'] == 'leaky':
        x = tf.keras.layers.LeakyReLU()(x)
    elif hparams['bottleneck_activation'] == 'gelu':
        x = tfa.layers.GELU()(x)
    x = tf.keras.layers.Dense(16 * 2 * hparams['output_seq_length'])(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], 16 * 2))(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv1D(256, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv1D(512, kernel_size=(hparams['kernel_size'],), strides=hparams['stride'], activation='relu',
                               padding='same', use_bias=False)(x)
    out = tf.keras.layers.Dense(1)(x)

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


def convolutional_4_layer_2_input(hparams, metrics):
    inp1 = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    inp2 = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x1 = tf.keras.layers.Conv1D(128, kernel_size=(3,), activation='relu', padding='same')(inp1)
    x2 = tf.keras.layers.Conv1D(128, kernel_size=(3,), activation='relu', padding='same')(inp2)
    x1 = tf.keras.layers.Conv1D(64, kernel_size=(3,), activation='relu', padding='same')(x1)
    x2 = tf.keras.layers.Conv1D(64, kernel_size=(3,), activation='relu', padding='same')(x2)
    x1 = tf.keras.layers.Conv1D(16, kernel_size=(3,), activation='relu', padding='same')(x1)
    x2 = tf.keras.layers.Conv1D(16, kernel_size=(3,), activation='relu', padding='same')(x2)
    x = tf.keras.layers.concatenate([x1, x2])
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['bottleneck_size'])(x)
    if hparams['bottleneck_activation'] == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif hparams['bottleneck_activation'] == 'leaky':
        x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(16 * 2 * hparams['output_seq_length'])(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], 16 * 2))(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=(3,), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=(3,),activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=(3,),activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv1D(256, kernel_size=(3,),activation='relu', padding='same')(x)
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs={'x1': inp1, 'x2': inp2}, outputs=[out])

    model.compile(loss=hparams['loss_function'], optimizer='adam', metrics=metrics)

    return model
