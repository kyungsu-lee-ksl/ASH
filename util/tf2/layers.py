import tensorflow as tf


class ASH(tf.keras.layers.Layer):

    def __init__(self):
        super(ASH, self).__init__()
        self.k = self.add_weight(shape=(1,), initializer="zeros", trainable=True)
        self.a = self.add_weight(shape=(1,), initializer="ones", trainable=True)

    def call(self, inputs):
        return tf.keras.activations.swish(-2 * self.a * (inputs - tf.math.reduce_mean(inputs) - self.k * tf.math.reduce_std(inputs)))


class Swish(tf.keras.layers.Layer):

    def __init__(self):
        super(Swish, self).__init__()

    def call(self, inputs):
        return tf.keras.activations.swish(inputs)



