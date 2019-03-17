import tensorflow as tf


def mask_with_inf(inputs, mask):
	"""Mask inputs tensor with a very large negtive number.
	
	Args:
		inputs: float. A tensor with shpe [batch_size, seq_len, depth]
		mask: int. A tensor with shape [batch_size, seq_len]
		
	Retuens:
		A tensor with same shape and data type as 'inputs'.
	"""
	adder = (1.0 - tf.cast(tf.expand_dims(mask, axis=-1), tf.float32)) * (-10000.0)
	outputs = inputs + adder
	return outputs
	
def mask_with_matmul(inputs, mask):
	"""Mask inputs tensor with 'matmul' op.
	
	Args:
		inputs: float. A tensor with shpe [batch_size, seq_len, depth]
		mask: int. A tensor with shape [batch_size, seq_len]
		
	Retuens:
		A tensor with same shape and data type as 'inputs'.
	"""
	outputs = inputs * tf.expand_dims(mask, axis=-1)
	return outputs
