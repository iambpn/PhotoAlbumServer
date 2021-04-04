from flask import Flask, request, render_template, redirect, url_for
import json
from PIL import Image
import io
import base64

from AI_Model.Prediction import greedyPrediction
from AI_Model.ImageEncoder import encodeImage, getVggModel
from AI_Model.CaptionDecoder import getCaptionModel
from AI_Model.Load_Tokenization import loadTokenization

app = Flask(__name__)
supported_type = ['image/png', 'image/jpeg']
word_to_idx, idx_to_word = None, None
max_length = 37
captionModelPath = "AI_Model/Model/model-val_loss2.858.h5"
word_to_idx_path = "AI_Model/Tokenization/word_index_37.pkl"
idx_to_word_path = "AI_Model/Tokenization/index_word_37.pkl"


@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")


@app.route('/get-prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':

        string_encoded_image: str = request.form.get("encodedImage", None)
        if string_encoded_image is not None:

            # May cause exception while converting in image string encoded data is not passed
            try:
                image = Image.open(io.BytesIO(base64.decodebytes(string_encoded_image.encode())))
            except Exception as ex:
                print(ex)
                return {"Error": "Data not Recognized. \n This exception is caused by " + str(ex)}, 400

            image_vector = encodeImage(image)
            cap_model = getCaptionModel(captionModelPath)
            caption, _ = greedyPrediction(cap_model, word_to_idx, idx_to_word, image_vector, max_length)

            # clean Caption
            caption = caption.replace("startseq", "").replace("endseq", "").strip()

            # success return
            print("Success")
            return json.dumps(caption), 200

        else:
            # if string_encoded_image is None
            print("Empty File")
            return {"Error": "File not selected. Empty file request!"}, 400


@app.errorhandler(404)
@app.errorhandler(405)
def exception(ex):
    if request.method == 'POST':
        return {"Error": "End Point Not Found"}, 404
    else:
        return render_template('404.html')


# loading Word_to_index and index_to_word
word_to_idx, idx_to_word = loadTokenization(word_to_idx_path, idx_to_word_path)

# Loading model before prediction
getVggModel()
getCaptionModel(captionModelPath)
print("Model Loaded\n")

if __name__ == '__main__':
    # start app
    app.run()

