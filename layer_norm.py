import tensorflow as tf

def layer_norm(inputs, epsilon=1e-8, scope=None, reuse=None):
    """Layer Normalization.
       https://arxiv.org/pdf/1607.06450.pdf

    Args:
        inputs: A tensor with 2 or more dimensions, where the first dimension has 'batch_size' (batch major).
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optinal scope for 'variable_scope'.
        reuse: Boolean, whether to reuse the weights of a previous layer by the same name.

    Returns:
        A tensor with the same shape and data dtype as 'inputs'.
    """
    with tf.variable_scope(scope or 'layer_norm', reuse=reuse):
        # The axes for centering and scaling.
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]    # Perform centering and scaling over last dimension.

        # Compute the 'mean' and 'variance' for Normalization along last dimension.
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        # Create the trainable params 'gamma' and 'beta' for centering and scaling.
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))

        # Normalizing.
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)

        # Scaling and recentering.
        outputs = gamma * normalized + beta

    return outputs
    
