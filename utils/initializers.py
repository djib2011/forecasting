import tensorflow as tf


class VarianceScalingV2(tf.keras.initializers.VarianceScaling):

    def __init__(self, std=1.0,
                 variance_scale=1.0,
                 mode="fan_in",
                 distribution="truncated_normal",
                 seed=None):
        super(VarianceScalingV2, self).__init__(scale=variance_scale, mode=mode,
                                                distribution=distribution,
                                                seed=seed)

        self.std = std

    def __call__(self, shape, dtype=tf.float32):
        out = super(VarianceScalingV2, self).__call__(shape, dtype=dtype)
        scaled_out = self.std * out
        return scaled_out
