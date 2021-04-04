import tensorflow as tf

captionModel = None

def getCaptionModel(model_path):
    global captionModel
    if captionModel is None:
        captionModel = tf.keras.models.load_model(model_path)
        return captionModel
    else:
        return captionModel
