import tensorflow as tf
import utils


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


def sha_rnn(hparams, metrics):

    return


# class SHARNN(tf.keras.Model):
#
#     def __init__(self, num_token, embed_dim, num_hid, num_layers,
#                  dropout=0.5, dropout_h=0.5, dropout_i=0.5,
#                  return_hidden=False, return_mem=False):
#         super(SHARNN, self).__init__()
#
#         num_embeddings = num_token
#         embed_dim = embed_dim
#         hidden_dim = num_hid
#
#         self.num_inp = embed_dim
#         self.num_hidden = num_hid
#         self.num_layers = num_layers
#
#         self.num_max_positions = 5000
#         self.num_heads = 1
#         self.causal = True
#
#         self.dropout = tf.keras.layers.Dropout(dropout)
#         self.i_dropout = tf.keras.layers.Dropout(dropout_i)
#
#         self.blocks = []
#         for idx in range(num_layers):
#             rnn = True
#             block = utils.layers.SHARNNBlock(embed_dim, hidden_dim, self.num_heads, dropout_h, rnn=rnn, residual=False,
#                                              use_attn=True if idx == num_layers - 2 else False)
#
#             self.blocks.append(block)
#
#         self.pos_embedding = [0] * self.num_max_positions
#         # self.decoder = tf.keras.layers.Dense(num_embeddings)
#
#         self.return_hidden = return_hidden
#         self.return_mem = return_mem
#
#     def call(self, inputs, hidden=None, mems=None, training=None, mask=None):
#         """ Input has shape [batch, seq length] """
#         e = self.encoder(inputs)
#         e = self.i_dropout(e, training=training)
#
#         batchsize = tf.shape(inputs)[0]
#         in_seqlen = tf.shape(inputs)[1]
#         out_seqlen = tf.shape(e)[1]
#
#         positional_encoding = self.pos_embedding
#         h = e
#
#         new_hidden = []
#         new_mems = []
#
#         attn_mask = tf.ones([in_seqlen, in_seqlen])
#         attn_mask = 1. - tf.linalg.band_part(attn_mask, -1, 0)
#
#         if mems:
#             m_shapes = [tf.shape(m) for m in mems]
#             m_seqlen = [m[1] if len(m) > 1 else m[0] for m in m_shapes]
#             max_mems = tf.reduce_max(m_seqlen)
#
#             happy = tf.zeros([in_seqlen, max_mems])
#             attn_mask = tf.concat([happy, attn_mask], axis=-1)
#
#         for idx, block in enumerate(self.blocks):
#             mem = mems[idx] if mems else None
#             hid = hidden[idx] if hidden else None
#             h, m, nh, f = block(h, positional_encoding, attn_mask, mem=mem, hidden=hid, training=training)
#
#             new_hidden.append(nh)
#             new_mems.append(m)
#
#         h = self.dropout(h, training=training)
#
#         output = [h]
#
#         if self.return_hidden:
#             output.append(new_hidden)
#
#         if self.return_mem:
#             output.append(new_mems)
#
#         if len(output) == 1:
#             output = output[0]
#
#         return output
