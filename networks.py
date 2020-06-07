import tensorflow as tf


def unidirectional_ae_2_layer(hparams, metrics):
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.LSTM(64, return_sequences=True, activation='relu')(inp)
    x = tf.keras.layers.LSTM(16, return_sequences=True, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['bottleneck_size'])(x)
    if hparams['bottleneck_activation'] == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif hparams['bottleneck_activation'] == 'leaky':
        x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(16 * 2 * hparams['output_seq_length'])(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], 16 * 2))(x)
    x = tf.keras.layers.LSTM(16, return_sequences=True, activation='relu')(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True, activation='relu')(x)
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inp, out)

    model.compile(loss=hparams['loss_function'], optimizer='adam', metrics=metrics)

    return model


def bidirectional_ae_2_layer(hparams, metrics):
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True, activation='relu'))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['bottleneck_size'])(x)
    if hparams['bottleneck_activation'] == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif hparams['bottleneck_activation'] == 'leaky':
        x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(16 * 2 * hparams['output_seq_length'])(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], 16*2))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True, activation='relu'))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss=hparams['loss_function'], optimizer='adam', metrics=metrics)
    return model


def bidirectional_ae_3_layer(hparams, metrics):
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, activation='relu'))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True, activation='relu'))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['bottleneck_size'])(x)
    if hparams['bottleneck_activation'] == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif hparams['bottleneck_activation'] == 'leaky':
        x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(16 * 2 * hparams['output_seq_length'])(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], 16*2))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True, activation='relu'))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, activation='relu'))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss=hparams['loss_function'], optimizer='adam', metrics=metrics)
    return model


def convolutional_ae_2_layer(hparams, metrics):
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Conv1D(64, kernel_size=(3,), activation='relu', padding='same')(inp)
    x = tf.keras.layers.Conv1D(16, kernel_size=(3,), activation='relu', padding='same')(x)
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
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inp, out)

    model.compile(loss=hparams['loss_function'], optimizer='adam', metrics=metrics)

    return model


def convolutional_ae_3_layer(hparams, metrics):
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Conv1D(128, kernel_size=(3,), activation='relu', padding='same')(inp)
    x = tf.keras.layers.Conv1D(64, kernel_size=(3,), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=(3,), activation='relu', padding='same')(x)
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
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inp, out)

    model.compile(loss=hparams['loss_function'], optimizer='adam', metrics=metrics)

    return model


def fully_connected_ae_2_layer(hparams, metrics):
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Dense(64, activation='relu')(inp)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['bottleneck_size'])(x)
    if hparams['bottleneck_activation'] == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif hparams['bottleneck_activation'] == 'leaky':
        x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(16 * 2 * hparams['output_seq_length'])(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], 16 * 2))(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inp, out)

    model.compile(loss=hparams['loss_function'], optimizer='adam', metrics=metrics)

    return model


def combined_ae_2_layer(hparams, metrics):
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, activation='relu'))(inp)
    x2 = tf.keras.layers.Conv1D(64, kernel_size=(3,), activation='relu', padding='same')(inp)
    x = tf.keras.layers.add([x1, x2])
    x1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True, activation='relu'))(x)
    x2 = tf.keras.layers.Conv1D(16, kernel_size=(3,), activation='relu', padding='same')(x)
    x = tf.keras.layers.add([x1, x2])
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['bottleneck_size'])(x)
    if hparams['bottleneck_activation'] == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif hparams['bottleneck_activation'] == 'leaky':
        x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(16 * 2 * hparams['output_seq_length'])(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], 16*2))(x)
    x1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True, activation='relu'))(x)
    x2 = tf.keras.layers.Conv1D(16, kernel_size=(3,), activation='relu', padding='same')(x)
    x = tf.keras.layers.add([x1, x2])
    x1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, activation='relu'))(x)
    x2 = tf.keras.layers.Conv1D(64, kernel_size=(3,),activation='relu', padding='same')(x)
    x = tf.keras.layers.add([x1, x2])
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss=hparams['loss_function'], optimizer='adam', metrics=metrics)
    return model
