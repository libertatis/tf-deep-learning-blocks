import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


def positional_embed(batch_size,
                     seq_length,
                     max_seq_len,
                     num_units,
                     zero_pad=True,
                     scale=True,
                     scope='positional_embed',
                     reuse=None):
    """Positional embedding.

    max_seq_len <==> vocabulary
    seq_length <==> sequence

    embedding_matrix : [max_seq_len, embedding_size]

    """
    with tf.variable_scope(scope, reuse=reuse):
        # [batch_size, seq_length]
        pos_seq = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])

        # [max_seq_len, num_units]
        lookup_table = tf.get_variable(
            name='lookup_table',
            dtype=tf.float32,
            shape=[max_seq_len, num_units],
            initializer=xavier_initializer())

        if zero_pad:
            lookup_table = tf.concat([tf.zeros(shape=[1, num_units]), lookup_table[1:, :]], axis=0)

        # [batch_size, seq_length, num_units]
        outputs = tf.nn.embedding_lookup(lookup_table, pos_seq)

        if scale:
            outputs = outputs * np.sqrt(num_units)

        return outputs

def sinusoidal_positional_embed(batch_size,
                                seq_length,
                                max_seq_length,
                                num_units,
                                zero_pad=True,
                                scale=True,
                                scope='sinusoidal_positional_embed',
                                reuse=None):
    """Sinusoidal positional embedding.

    """
    with tf.variable_scope(scope, reuse=reuse):
        # [batch_size, seq_length]
        pos_seq = tf.tile(tf.expand_dims(tf.range(seq_length), axis=0), [batch_size, 1])

        # First part of the PE function: sin and cos argument
        # [max_seq_length, num_units]
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(max_seq_length)
        ])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        # Convert to a tensor [max_seq_len, num_units]
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat([tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]], axis=0)

        # [batch_size, seq_length, num_units]
        outputs = tf.nn.embedding_lookup(lookup_table, pos_seq)

        if scale:
            outputs = outputs * np.sqrt(num_units)

        return outputs
