import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import Model


class ImageModel:
    __model = None

    def __init__(self):
        # loading early
        self.get_resnet_model()

    @staticmethod
    def __preprocess_image(image, target_size=(224, 224)):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target_size)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, 0)
        return image

    @staticmethod
    def get_resnet_model():
        if ImageModel.__model is None:
            model = ResNet50(weights="imagenet")
            ImageModel.__model = Model(model.input, model.layers[-2].output)
            return ImageModel.__model
        else:
            return ImageModel.__model

    def encode_image(self, image):
        model = self.get_resnet_model()
        image_array = self.__preprocess_image(image, target_size=(224, 224))
        image_vector = model.predict(image_array)
        image_vector = image_vector.reshape(1, image_vector.shape[1])
        return image_vector
