from flask import Flask, request, render_template
import json
from PIL import Image
import io
import base64
from AI_Model.ImgCap import ImgCap

app = Flask(__name__)
supported_type = ['image/png', 'image/jpeg']
img_cap = None


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
                # to save the image
                #open("image.jpg", "wb").write(base64.decodebytes(string_encoded_image.encode()))
            except Exception as ex:
                print(ex)
                return {"Error": "Data not Recognized. \n This exception is caused by " + str(ex)}, 400

            caption = img_cap.get_image_caption(image)

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


if img_cap is None:
    img_cap = ImgCap()

if __name__ == '__main__':
    # start app
    app.run()
