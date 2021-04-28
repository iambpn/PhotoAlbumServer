from .ImageEncoder import ImageModel
from .CaptionDecoder import CaptionModel


class ImgCap:
    def __init__(self):
        self.imgModel = ImageModel()
        self.capModel = CaptionModel()

    def get_image_caption(self, image):
        image_vector = self.imgModel.encode_image(image)
        caption, _ = self.capModel.greedy_prediction(image_vector)
        # clean Caption
        caption = caption.replace("startseq", "").replace("endseq", "").strip()
        return caption
