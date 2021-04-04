from Prediction import greedyPrediction
from ImageEncoder import encodeImage, getVggModel
from CaptionDecoder import getCaptionModel
from Load_Tokenization import loadTokenization
from PIL import Image

max_length = 37
captionModelPath = "Model/model-val_loss2.858.h5"

word_to_idx, idx_to_word = loadTokenization("Tokenization/word_index_37.pkl",
                                            "Tokenization/index_word_37.pkl")

image = Image.open("test2.jpg")
image_vector = encodeImage(image)
cap_model = getCaptionModel(captionModelPath)
caption, _ = greedyPrediction(cap_model, word_to_idx, idx_to_word, image_vector, max_length)

print(caption)