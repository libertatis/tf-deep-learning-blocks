import numpy as np
import tensorflow as tf

def gelu(x):
    """Gaussian Error Linear Unit (GELU)
        
        GAUSSIAN ERROR LINEAR UNITS (GELUS)
        Dan Hendrycks and Kevin Gimpel
        https://arxiv.org/pdf/1606.08415.pdf

    This is a smoother version of the ReLU.

    Args:
        x: float Tensor to perform activation.

    Returns:
        'x' with the gelu activation applied.
    """

    return 0.5 * x * (1 + tf.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def relu(x):
    return tf.maximum(x, 0)

def leaky_relu(x, leak=0.2, name='lrelu'):
    """Compute the Leaky ReLU activation function.

        Rectifier Nonlinearities Improve Neural Network Acoustic Models
        AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
        https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf

    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def crelu(x, name=None, axis=-1):
    """Computes Concatenated ReLU.

        Understanding and Improving Convolutional Neural Networks via
        Concatenated Rectified Linear Units. W. Shang, etal.
        https://arxiv.org/abs/1603.05201

    """
    with tf.name_scope('crelu', [x]) as name:
        x = tf.convert_to_tensor(x, name='x')
        cat = tf.concat([x, -x], axis=axis, name=name)
        return relu(cat, 0)

def elu(x, alpha=1.0):
    """Computes Exponential Linear Units, ELUs.

        Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
        Djork-Arne Clevert, Thomas Unterthiner & Sepp Hochreiter
        https://arxiv.org/pdf/1511.07289.pdf

    """
    return tf.maximum(x, alpha*(tf.exp(x) - 1))

def selu(x):
    """Computes Scaled Exponential Linear Units, SELUs.

        Self-Normalizing Neural Networks
        Günter Klambauer, Thomas Unterthiner, Andreas Mayr and Sepp Hochreiter
        https://arxiv.org/pdf/1706.02515.pdf
        https://github.com/bioinf-jku/SNNs/blob/master/selu.py
    
    """
    with tf.name_scope('selu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

