import tensorflow as tf
import tensorflow_addons as tfa
import utils


class EncoderBlock(tf.keras.layers.Layer):

    def __init__(self, layer_type='conv', num_layers=3, kernel_size=5):
        """
        Sequence-to-sequence encoder block.

        'layer_type' determines what type of a sequential model we'll have, while 'num_layers' determines how deep the
        model will be. For example if layer_type='conv' and num_layers=3, then the encoder would be equivalent to:

        ```python
        x = tf.keras.layers.Conv1D(128)(x)
        x = tf.keras.layers.Conv1D(64)(x)
        x = tf.keras.layers.Conv1D(32)(x)
        ```

        :param layer_type: The type of the layers. Three options are available: 'conv', 'lstm' and 'bi'.
        :param num_layers: The number of layers in the encoder.
        :param kernel_size: The kernel size of convolutional layers. Only relevant if layer_type='conv'
        """

        super(EncoderBlock, self).__init__()

        if layer_type == 'conv':
            base_layer = lambda x: tf.keras.layers.Conv1D(x, kernel_size=kernel_size, padding='same',
                                                          activation='relu', strides=1, use_bias=False)
        elif layer_type == 'lstm':
            base_layer = lambda x: tf.keras.layers.LSTM(x, return_sequences=True, activation='relu')
        elif layer_type == 'bi':
            base_layer = lambda x: tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(x, return_sequences=True,
                                                                                      activation='relu'))
        else:
            raise ValueError()

        sizes = [512, 256, 128, 64, 32]
        sizes = sizes[-num_layers:]

        self._layers_builder = [base_layer(sizes[i]) for i in range(num_layers)]

    def __call__(self, x, *args, **kwargs):

        y_ = x
        for layer in self._layers_builder:
            y_ = layer(y_)

        return y_


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, layer_type='conv', num_layers=3, kernel_size=5):
        """
        Sequence-to-sequence decoder block.

        'layer_type' determines what type of a sequential model we'll have, while 'num_layers' determines how deep the
        model will be. For example if layer_type='conv' and num_layers=3, then the encoder would be equivalent to:

        ```python
        x = tf.keras.layers.Conv1D(32)(x)
        x = tf.keras.layers.Conv1D(64)(x)
        x = tf.keras.layers.Conv1D(128)(x)
        ```

        :param layer_type: The type of the layers. Three options are available: 'conv', 'lstm' and 'bi'.
        :param num_layers: The number of layers in the encoder.
        :param kernel_size: The kernel size of convolutional layers. Only relevant if layer_type='conv'
        """

        super(DecoderBlock, self).__init__()

        if layer_type == 'conv':
            base_layer = lambda x: tf.keras.layers.Conv1D(x, kernel_size=kernel_size, padding='same',
                                                          activation='relu', strides=1, use_bias=False)
        elif layer_type == 'lstm':
            base_layer = lambda x: tf.keras.layers.LSTM(x, return_sequences=True, activation='relu')
        elif layer_type == 'bi':
            base_layer = lambda x: tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(x, return_sequences=True,
                                                                                      activation='relu'))
        else:
            raise ValueError()

        sizes = [32, 64, 128, 256, 512]
        sizes = sizes[:num_layers]

        self._layers_builder = [base_layer(sizes[i]) for i in range(num_layers)]

    def __call__(self, x, *args, **kwargs):

        y_ = x
        for layer in self._layers_builder:
            y_ = layer(y_)

        return y_


class AttentionBottleneck(tf.keras.layers.Layer):

    def __init__(self, output_length, bottleneck_size, layer_type='conv', scale=True, kernel_size=5,
                 bottleneck_activation='relu'):
        """
        Sequence-to-sequence bottleneck with attention.

        'layer_type' determines what type of a sequential model we'll have, while 'num_layers' determines how deep the
        model will be. For example if layer_type='conv' and num_layers=3, then the encoder would be equivalent to:

        ```python
        x = tf.keras.layers.Conv1D(32)(x)
        x = tf.keras.layers.Conv1D(64)(x)
        x = tf.keras.layers.Conv1D(128)(x)
        ```

        :param output_length: The target length of the output sequence.
        :param bottleneck_size: The size of the FC part of the bottleneck.
        :param layer_type: The type of the layers. Three options are available: 'conv', 'lstm' and 'bi'.
        :param scale: Determines whether to use scaled or regular attention.
        :param kernel_size: The kernel size of convolutional layers. Only relevant if layer_type='conv'
        :param bottleneck_activation: The activation function of the bottleneck. Options: 'relu', 'leaky' and 'gelu'
        """


        super(AttentionBottleneck, self).__init__()

        if layer_type == 'conv':
            base_layer = lambda: tf.keras.layers.Conv1D(16, kernel_size=kernel_size, padding='same',
                                                          activation='relu', strides=1)
        elif layer_type == 'lstm':
            base_layer = lambda: tf.keras.layers.LSTM(16, return_sequences=True, activation='relu')
        elif layer_type == 'bi':
            base_layer = lambda: tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True,
                                                                                      activation='relu'))
        else:
            raise ValueError()

        if bottleneck_activation == 'relu':
            activation = tf.keras.layers.ReLU()
        elif bottleneck_activation == 'leaky':
            activation = tf.keras.layers.LeakyReLU()
        elif bottleneck_activation == 'gelu':
            activation = tfa.layers.GELU()
        else:
            raise ValueError()

        flatten = tf.keras.layers.Flatten()
        bottleneck1 = tf.keras.layers.Dense(bottleneck_size)
        bottleneck2 = tf.keras.layers.Dense(16 * output_length)
        reshape = tf.keras.layers.Reshape((output_length, 16))
        self.attn = tf.keras.layers.Attention(use_scale=scale)

        self._layers_builder = [base_layer(), flatten, bottleneck1, activation, bottleneck2, reshape, base_layer()]

    def __call__(self, x, *args, **kwargs):

        value = self._layers_builder[0](x)
        y_ = value

        for layer in self._layers_builder[1:]:
            y_ = layer(y_)

        query = y_
        y_ = self.attn([query, value])

        return y_


class DisjoinedAttentionBottleneck(tf.keras.layers.Layer):

    def __init__(self, output_length, bottleneck_size, layer_type='conv', scale=True, kernel_size=5,
                 bottleneck_activation='relu'):
        """
        Sequence-to-sequence bottleneck with attention.

        'layer_type' determines what type of a sequential model we'll have, while 'num_layers' determines how deep the
        model will be. For example if layer_type='conv' and num_layers=3, then the encoder would be equivalent to:

        ```python
        x = tf.keras.layers.Conv1D(32)(x)
        x = tf.keras.layers.Conv1D(64)(x)
        x = tf.keras.layers.Conv1D(128)(x)
        ```

        :param output_length: The target length of the output sequence.
        :param bottleneck_size: The size of the FC part of the bottleneck.
        :param layer_type: The type of the layers. Three options are available: 'conv', 'lstm' and 'bi'.
        :param scale: Determines whether to use scaled or regular attention.
        :param kernel_size: The kernel size of convolutional layers. Only relevant if layer_type='conv'
        :param bottleneck_activation: The activation function of the bottleneck. Options: 'relu', 'leaky' and 'gelu'
        """


        super(DisjoinedAttentionBottleneck, self).__init__()

        if layer_type == 'conv':
            base_layer = lambda: tf.keras.layers.Conv1D(16, kernel_size=kernel_size, padding='same',
                                                          activation='relu', strides=1)
        elif layer_type == 'lstm':
            base_layer = lambda: tf.keras.layers.LSTM(16, return_sequences=True, activation='relu')
        elif layer_type == 'bi':
            base_layer = lambda: tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True,
                                                                                      activation='relu'))
        else:
            raise ValueError()

        if bottleneck_activation == 'relu':
            activation = tf.keras.layers.ReLU()
        elif bottleneck_activation == 'leaky':
            activation = tf.keras.layers.LeakyReLU()
        elif bottleneck_activation == 'gelu':
            activation = tfa.layers.GELU()
        else:
            raise ValueError()

        flatten = tf.keras.layers.Flatten()
        bottleneck1 = tf.keras.layers.Dense(bottleneck_size)
        bottleneck2 = tf.keras.layers.Dense(16 * output_length)
        reshape = tf.keras.layers.Reshape((output_length, 16))
        self.attn = tf.keras.layers.Attention(use_scale=scale)

        self._layers_builder = [base_layer(), flatten, bottleneck1, activation, bottleneck2, reshape, base_layer()]

    def __call__(self, x, *args, **kwargs):

        value = self._layers_builder[0](x)
        y_ = value

        for layer in self._layers_builder[1:]:
            y_ = layer(y_)

        query = y_
        y_ = self.attn([query, value])

        return y_


'''
if False:
    class Attention(tf.keras.layers.Layer):

        def __init__(self, num_hidden, num_heads, q=True, k=False, v=False, r=False, dropout=None):
            """
            Source: https://github.com/titu1994/tf-sha-rnn/blob/master/sharnn.py

            :param num_hidden:
            :param num_heads:
            :param q:
            :param k:
            :param v:
            :param r:
            :param dropout:
            """
            super(Attention, self).__init__()

            assert num_hidden % num_heads == 0, 'Heads must divide vector evenly'

            self.num_hidden = num_hidden
            self.num_heads = num_heads
            self.depth = num_hidden // num_heads

            self.qs = tf.Variable(tf.zeros([1, 1, num_hidden]))
            self.ks = tf.Variable(tf.zeros([1, 1, num_hidden]))
            self.vs = tf.Variable(tf.zeros([1, 1, num_hidden]))

            self.qkvs = tf.Variable(tf.zeros([1, 3, num_hidden]))

            self.dropout = tf.keras.layers.Dropout(dropout) if dropout is not None else tf.keras.layers.Dropout(0.)
            self.gelu = tfa.layers.GELU()

            self.q = tf.keras.layers.Dense(num_hidden, kernel_initializer=utils.initializers.VarianceScalingV2(0.1)) if q else None
            self.k = tf.keras.layers.Dense(num_hidden, kernel_initializer=utils.initializers.VarianceScalingV2(0.1)) if k else None
            self.v = tf.keras.layers.Dense(num_hidden, kernel_initializer=utils.initializers.VarianceScalingV2(0.1)) if v else None
            self.r = tf.keras.layers.Dense(num_hidden, kernel_initializer=utils.initializers.VarianceScalingV2(0.1)) if r else None
            self.r_gate = tf.Variable(tf.ones([1, 1, num_hidden]))

            self.qln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                          gamma_initializer=utils.initializers.VarianceScalingV2(0.1))

        def split_heads(self, x, batch_size):
            """
            Split the last dimension into (num_heads, depth).
            Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
            """
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def call(self, query, key, value, training=None, mask=None):
            qs = tf.nn.sigmoid(self.qs)
            ks = tf.nn.sigmoid(self.ks)
            vs = tf.nn.sigmoid(self.vs)

            query = self.q(query)
            query = self.qln(query)

            key = self.k(key)

            value = self.v(value)

            q = qs * query
            k = ks * key
            v = vs * value

            q = self.dropout(q, training=training)
            v = self.dropout(v, training=training)

            original_q = tf.identity(q)

            batch_size = tf.shape(q)[0]
            q = self.split_heads(q, batch_size)
            k = self.split_heads(k, batch_size)
            v = self.split_heads(v, batch_size)

            mix, focus = self.scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout, training=training)
            mix = tf.transpose(mix, [0, 2, 1, 3])
            mix = tf.reshape(mix, [batch_size, -1, self.num_hidden])

            if self.r:
                r = tf.concat([mix, original_q], axis=-1)
                r = self.dropout(r, training=training)
                r = self.gelu(self.r(r))
                mix = tf.nn.sigmoid(self.r_gate) * mix + r

            return mix, focus

        @staticmethod
        def scaled_dot_product_attention(q, k, v, mask=None, dropout=None, training=None):
            """
            Calculate the attention weights.
            q, k, v must have matching leading dimensions.
            k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
            The mask has different shapes depending on its type(padding or look ahead)
            but it must be broadcastable for addition.
            Args:
              q: query shape == (..., seq_len_q, depth)
              k: key shape == (..., seq_len_k, depth)
              v: value shape == (..., seq_len_v, depth_v)
              mask: Float tensor with shape broadcastable
                    to (..., seq_len_q, seq_len_k). Defaults to None.
              dropout: Either a Dropout layer or None. If Dropout layer is provider,
                    ensure to pass `training` flag as well.
            Returns:
              output, attention_weights
            """
            matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

            # scale matmul_qk
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

            # add the mask to the scaled tensor.
            if mask is not None:
                scaled_attention_logits += (mask * -1e9)

            # softmax is normalized on the last axis (seq_len_k) so that the scores
            # add up to 1.
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

            if dropout is not None:
                attention_weights = dropout(attention_weights, training=training)

            output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

            return output, attention_weights


    class SHARNNBlock(tf.keras.layers.Layer):

        def __init__(self, inp_dim, hidden_dim, heads=1, dropout=None, rnn=False, residual=True, use_attn=True):
            super(SHARNNBlock, self).__init__()

            self.attn = None
            if use_attn:
                self.attn = Attention(inp_dim, num_heads=heads, r=False, dropout=dropout)

            self.ln_start = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                               gamma_initializer=utils.initializers.VarianceScalingV2(0.1))
            self.ln_mid = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                             gamma_initializer=utils.initializers.VarianceScalingV2(0.1))
            self.ln_mem = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                             gamma_initializer=utils.initializers.VarianceScalingV2(0.1))
            self.ln_out = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                             gamma_initializer=utils.initializers.VarianceScalingV2(0.1))
            self.ln_ff = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                            gamma_initializer=utils.initializers.VarianceScalingV2(0.1))
            self.ln_xff = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                             gamma_initializer=utils.initializers.VarianceScalingV2(0.1))
            self.dropout = tf.keras.layers.Dropout(dropout) if dropout is not None else tf.keras.layers.Dropout(0.)
            self.gelu = tfa.layers.GELU()
            self.residual = residual

            self.rnn = None
            if rnn:
                self.rnn = tf.keras.layers.LSTM(inp_dim, return_state=True, return_sequences=True)

        def call(self, h, positional_encoding, attention_mask, mem=None, hidden=None, training=None):
            new_mem = None

            h = self.ln_start(h)

            if self.rnn:
                out = self.rnn(h, training=training, initial_state=hidden)
                x, hidden = out[0], out[1:]

                num_inp = tf.shape(h)[-1]
                x_shape = tf.shape(x)
                num_out = x_shape[-1]
                clip_out = num_out // num_inp * num_inp

                z = x[..., 0:clip_out]
                z_shape = tf.concat([x_shape[:-1], [num_out // num_inp, num_inp]], axis=0)
                z = tf.reshape(z, z_shape)

                x = self.dropout(z, training=training)
                x = tf.reduce_sum(x, axis=-2)

                if self.residual:
                    h = h + x
                else:
                    h = x

            focus = None
            new_mem = []

            if self.attn is not None:
                mh = self.ln_mem(h)
                h = self.ln_mid(h)

                if mem is not None:
                    bigh = tf.concat([mem, mh], axis=0)
                else:
                    bigh = mh

                new_mem = bigh[-tf.shape(positional_encoding)[0]:]

                q, k = h, bigh
                x, focus = self.attn(q, k, bigh, mask=attention_mask, training=training)
                x = self.dropout(x, training=training)
                h = x + h

            return h, new_mem, hidden, focus


    class TSBlock(tf.keras.layers.Layer):

        def __init__(self, inp_dim, dropout=0.):
            super(TSBlock, self).__init__()

            self.ln_start = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                               gamma_initializer=utils.initializers.VarianceScalingV2(0.1))
            self.ln_mid = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                             gamma_initializer=utils.initializers.VarianceScalingV2(0.1))
            self.ln_mem = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                             gamma_initializer=utils.initializers.VarianceScalingV2(0.1))


            self.rnn = tf.keras.layers.LSTM(inp_dim, return_state=True, return_sequences=True)
            self.dropout = tf.keras.layers.Dropout(dropout)

        def __call__(self, h, mem=None, hidden=None):

            h = self.ln_start(h)
            out = self.rnn(h, initial_state=hidden)
            x, hidden = out[0], out[1:]
            x = self.dropout(x)

            h = x

            mh = self.ln_mem(h)
            h = self.ln_mid(h)

            bigh = tf.concat([mem, mh], axis=0)

            new_mem = bigh[-tf.shape(positional_encoding)[0]:]

            q, k, v = h, bigh, bigh
            x, focus = self.attn(q, k, bigh, mask=attention_mask, training=training)
            x = self.dropout(x, training=training)
            h = x + h


    class Attention2(keras.layers.Layer):
        def __init__(self, step_dim, **kwargs):
            self.step_dim = step_dim
            super(Attention2, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(shape=(input_shape[-1],),
                                     initializer=keras.initializers.get('glorot_uniform'),
                                     name='{}_W'.format(self.name))

            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name))

            self.features_dim = input_shape[-1]

            super(Attention2, self).build(input_shape)

        def call(self, x, mask=None):
            features_dim = self.features_dim
            step_dim = self.step_dim

            eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                                  K.reshape(self.W, (features_dim, 1))),
                            (-1, step_dim)) + self.b

            a = K.exp(K.tanh(eij))

            # apply mask after the exp. will be re-normalized next
            if mask:
                a *= K.cast(mask, K.floatx())

            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

            a = K.expand_dims(a)
            weighted_input = x * a

            return K.sum(weighted_input, axis=1)

        def compute_output_shape(self, input_shape):
            return input_shape[0], self.features_dim
'''
