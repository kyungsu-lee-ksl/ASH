import tensorflow as tf

def VGG11(inputs, name='vgg', activation=tf.nn.relu, activation2=tf.nn.relu, num_classes=100):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        layer01 = conv(inputs, int(inputs.shape[3]), 64, name='vgg_layer_01', activation=activation)
        layer02 = max_pooling(layer01, name='vgg_layer_02')

        layer03 = conv(layer02, 64, 128, name='vgg_layer_03', activation=activation)
        layer04 = max_pooling(layer03, name='vgg_layer_04')

        layer05 = conv(layer04, 128, 256, name='vgg_layer_05', activation=activation)
        layer06 = conv(layer05, 256, 256, name='vgg_layer_06', activation=activation)
        layer07 = max_pooling(layer06, name='vgg_layer_07')

        layer08 = conv(layer07, 256, 512, name='vgg_layer_08', activation=activation)
        layer09 = conv(layer08, 512, 512, name='vgg_layer_09', activation=activation)
        layer10 = max_pooling(layer09, name='vgg_layer_10')

        layer11 = conv(layer10, 512, 512, name='vgg_layer_11', activation=activation)
        layer12 = conv(layer11, 512, 512, name='vgg_layer_12', activation=activation)
        layer13 = max_pooling(layer12, name='vgg_layer_13')

        flatten = tf.layers.flatten(layer13)
        dense01 = tf.layers.dense(flatten, units=100, activation=activation2, name='dense01')
        dense02 = tf.layers.dense(dense01, units=100, activation=activation2, name='dense02')
        dense03 = tf.layers.dense(dense02, units=num_classes, activation=activation2, name='dense03')

        output = tf.nn.softmax(dense03)
    return output




def VGG19(inputs, name='vgg', activation=tf.nn.relu, activation2=tf.nn.relu, num_classes=100):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        layer01 = conv(inputs, int(inputs.shape[3]), 64, name='vgg_layer_01', activation=activation)
        layer02 = conv(layer01, 64, 64, name='vgg_layer_02', activation=activation)
        layer03 = max_pooling(layer02, name='vgg_layer_03')

        layer04 = conv(layer03, 64, 128, name='vgg_layer_04', activation=activation)
        layer05 = conv(layer04, 128, 128, name='vgg_layer_05', activation=activation)
        layer06 = max_pooling(layer05, name='vgg_layer_06')

        layer07 = conv(layer06, 128, 256, name='vgg_layer_07', activation=activation)
        layer08 = conv(layer07, 256, 256, name='vgg_layer_08', activation=activation)
        layer09 = conv(layer08, 256, 256, name='vgg_layer_09', activation=activation)
        layer19 = conv(layer09, 256, 256, name='vgg_layer_19', activation=activation)
        layer10 = max_pooling(layer19, name='vgg_layer_10')

        layer11 = conv(layer10, 256, 512, name='vgg_layer_11', activation=activation)
        layer12 = conv(layer11, 512, 512, name='vgg_layer_12', activation=activation)
        layer13 = conv(layer12, 512, 512, name='vgg_layer_13', activation=activation)
        layer20 = conv(layer13, 512, 512, name='vgg_layer_20', activation=activation)
        layer14 = max_pooling(layer20, name='vgg_layer_14')

        layer15 = conv(layer14, 512, 512, name='vgg_layer_15', activation=activation)
        layer16 = conv(layer15, 512, 512, name='vgg_layer_16', activation=activation)
        layer17 = conv(layer16, 512, 512, name='vgg_layer_17', activation=activation)
        layer21 = conv(layer17, 512, 512, name='vgg_layer_21', activation=activation)
        layer18 = max_pooling(layer21, name='vgg_layer_18')

        flatten = tf.layers.flatten(layer18)
        dense01 = tf.layers.dense(flatten, units=100, activation=activation2, name='dense01')
        dense02 = tf.layers.dense(dense01, units=100, activation=activation2, name='dense02')
        dense03 = tf.layers.dense(dense02, units=num_classes, activation=activation2, name='dense03')

        output = tf.nn.softmax(dense03)

    return output
