import tensorflow as tf


def build_mape(overlap=6):

    def MAPE(y_true, y_pred):
        target = tf.slice(y_true, [0, overlap, 0], [tf.shape(y_true)[0], tf.shape(y_true)[1] - overlap, tf.shape(y_true)[2]])
        prediction = tf.slice(y_pred, [0, overlap, 0], [tf.shape(y_true)[0], tf.shape(y_true)[1] - overlap, tf.shape(y_true)[2]])
        return tf.keras.losses.MAPE(target, prediction)

    return MAPE


def build_smape(overlap=6):

    def SMAPE(y_true, y_pred):

        y_true = tf.cast(y_true, dtype=tf.float32)

        target = tf.slice(y_true, [0, overlap, 0], [tf.shape(y_true)[0], tf.shape(y_true)[1] - overlap, tf.shape(y_true)[2]])
        prediction = tf.slice(y_pred, [0, overlap, 0], [tf.shape(y_true)[0], tf.shape(y_true)[1] - overlap, tf.shape(y_true)[2]])

        nom = tf.abs(target - prediction)
        denom = tf.abs(target) + tf.abs(prediction) + tf.keras.backend.epsilon()

        return 2 * tf.math.reduce_mean(nom / denom, axis=[1, 2]) * 100
    return SMAPE


def build_mase(overlap=6, frequency=1):

    assert overlap > 0, 'The targets should overlap with the input for at least 1 point.'

    def MASE_estimate(y_true, y_pred):

        y_true = tf.cast(y_true, dtype=tf.float32)

        target = tf.slice(y_true, [0, overlap, 0], [tf.shape(y_true)[0], tf.shape(y_true)[1] - overlap, tf.shape(y_true)[2]])
        prediction = tf.slice(y_pred, [0, overlap, 0], [tf.shape(y_true)[0], tf.shape(y_true)[1] - overlap, tf.shape(y_true)[2]])
        beginning_0 = tf.slice(y_true, [0, 0, 0], [tf.shape(y_true)[0], overlap - frequency, tf.shape(y_true)[2]])
        beginning_1 = tf.slice(y_true, [0, frequency, 0], [tf.shape(y_true)[0], overlap - 1, tf.shape(y_true)[2]])  # I don't understand why (-1)

        nom = tf.reduce_mean(tf.abs(target - prediction), axis=[1, 2])
        denom = tf.reduce_mean(tf.abs(beginning_1 - beginning_0), axis=[1, 2]) + tf.keras.backend.epsilon()

        return nom / denom

    return MASE_estimate


def build_reconstruction_loss(overlap=6):

    def reconstruction_loss(y_true, y_pred):
        target = tf.slice(y_true, [0, 0, 0], [tf.shape(y_true)[0], overlap, tf.shape(y_true)[2]])
        prediction = tf.slice(y_pred, [0, 0, 0], [tf.shape(y_true)[0], overlap, tf.shape(y_true)[2]])
        return tf.keras.metrics.mean_squared_error(target, prediction)

    return reconstruction_loss


def build_owa(overlap=6, frequency=1, smape_baseline=15.201, mase_baseline=1.685):

    def OWA_estimate(y_true, y_pred):

        target = tf.slice(y_true, [0, overlap, 0], [tf.shape(y_true)[0], tf.shape(y_true)[1] - overlap, tf.shape(y_true)[2]])
        prediction = tf.slice(y_pred, [0, overlap, 0], [tf.shape(y_true)[0], tf.shape(y_true)[1] - overlap, tf.shape(y_true)[2]])
        beginning_0 = tf.slice(y_true, [0, 0, 0], [tf.shape(y_true)[0], overlap - frequency, tf.shape(y_true)[2]])
        beginning_1 = tf.slice(y_true, [0, frequency, 0], [tf.shape(y_true)[0], overlap - 1, tf.shape(y_true)[2]])  # I don't understand why (-1)

        smape_nom = tf.abs(target - prediction)
        smape_denom = tf.abs(target) + tf.abs(prediction) + tf.keras.backend.epsilon()
        smape = 2 * tf.math.reduce_mean(smape_nom / smape_denom, axis=[1, 2]) * 100
        relative_smape = smape / smape_baseline

        mase_nom = tf.reduce_mean(tf.abs(target - prediction), axis=[1, 2])
        mase_denom = tf.reduce_mean(tf.abs(beginning_1 - beginning_0), axis=[1, 2]) + tf.keras.backend.epsilon()
        mase = mase_nom / mase_denom
        relative_mase = mase / mase_baseline

        return (relative_smape + relative_mase) / 2

    return OWA_estimate
