import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras import Model

vgg_model = None


def preprocessImage(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, 0)
    return image


def getVggModel():
    global vgg_model
    if vgg_model is None:
        vgg_model = VGG16(weights="imagenet")
        vgg_model = Model(vgg_model.input, vgg_model.layers[-2].output)
        return vgg_model
    else:
        return vgg_model


def encodeImage(image):
    model = getVggModel()
    image_array = preprocessImage(image, target_size=(224, 224))
    image_vector = np.reshape(model.predict(image_array), (4096,))
    image_vector = np.expand_dims(image_vector, axis=1)
    image_vector = image_vector.reshape((1, 4096))
    return image_vector
