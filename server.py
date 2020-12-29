import io
import json
from flask_cors import CORS
from flask import Flask, request, Response
from werkzeug.datastructures import FileStorage
from PIL import Image
from utilities import *

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict_image():
    file = request.files.get('image')  # type: FileStorage
    image = Image.open(file)
    image = np.array(image)

    results = Search().results(image)
    payload = json.dumps(results).encode()
    return Response(
        payload,
        status=200,
        mimetype="application/json"
    )


@app.route('/getImage', methods=['GET'])
def get_image():
    filename = request.args.get("name")
    filenamepath = os.path.join(os.getcwd(), constants.DATASET_IMAGES_FOLDER, filename)

    image = Image.open(filenamepath)
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format=image.format)
    image_byte_array = image_byte_array.getvalue()

    return Response(
        [image_byte_array],
        status=200,
        mimetype="image/jpeg")


@app.route('/getImage/<class_name>', methods=['GET'])
def get_image_by_class(class_name):
    filename = request.args.get("name")
    filenamepath = os.path.join(os.getcwd(), constants.CLASSIFIED_DATASET_IMAGES_FOLDER, class_name, filename)

    image = Image.open(filenamepath)
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format=image.format)
    image_byte_array = image_byte_array.getvalue()

    return Response(
        [image_byte_array],
        status=200,
        mimetype="image/jpeg")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
