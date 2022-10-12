import keras

from util.tf1 import ASH, VGG11, VGG19
from util.tf2 import ASH, Swish, insert_layer_nonseq



## Tensorflow 1
if __name__ == '__main__':

    imgHolder = tf.placeHolder(dtype=tf.float32, shape=(None, 256, 256, 3))

    model1 = VGG11(imgHolder, activation=ASH, activation2=tf.nn.relu, num_classes=1000, name='vgg11')
    model2 = VGG19(imgHolder, activation=ASH, activation2=tf.nn.relu, num_classes=1000, name='vgg19')



## Tensorflow 2 & Keras
if __name__ == '__main__':
    model_builder = keras.applications.xception.Xception
    img_size = (299, 299)

    # Make model
    model1 = model_builder(weights="imagenet")
    model1.layers[-1].activation = None
    model1 = insert_layer_nonseq(model1, '.*act', ASH)

    model2 = model_builder(weights="imagenet")
    model2.layers[-1].activation = None
    model2 = insert_layer_nonseq(model2, 'block13_sepconv2_act', Swish)

    model1.summary()
    model2.summary()
