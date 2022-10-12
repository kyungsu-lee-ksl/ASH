import tensorflow as tf



def ASH(x):
    moments = tf.nn.moments(x, axes=[1, 2, 3], keep_dims=True)
    mean, sigma = moments

    mean = tf.identity(mean)
    sigma = tf.identity(tf.sqrt(sigma))

    z_k = tf.get_variable(name='z_k', shape=(1,), trainable=True, initializer=tf.constant_initializer(1e-3))
    alpha = tf.get_variable(name='alpha', shape=(1,), trainable=True, initializer=tf.constant_initializer(1.0))

    return x / (1 + tf.math(-2 * alpha * (x - mean - z_k * sigma)))
