import io
import json
import os
import constants
from flask_cors import CORS
from flask import Flask, request, Response
import numpy as np
from werkzeug.datastructures import FileStorage
from PIL import Image
import search

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict_image():
    file = request.files.get('image')  # type: FileStorage
    image = Image.open(file)
    image = np.array(image)

    results = search.results(image)
    payload = json.dumps(results).encode()
    return Response(
        payload,
        status=200,
        mimetype="application/json"
    )


@app.route('/getImage', methods=['GET'])
def get_image():
    file_name = request.args.get("name")
    file_path = os.path.join(os.getcwd(), constants.DATASET_IMAGES_FOLDER, file_name)

    image = Image.open(file_path)
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format=image.format)
    image_byte_array = image_byte_array.getvalue()

    return Response(
        [image_byte_array],
        status=200,
        mimetype="image/jpeg")


@app.route('/getImage/<class_name>', methods=['GET'])
def get_image_by_class(class_name):
    file_name = request.args.get("name")
    file_path = os.path.join(os.getcwd(), constants.CLASSIFIED_DATASET_IMAGES_FOLDER, class_name, file_name)

    image = Image.open(file_path)
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format=image.format)
    image_byte_array = image_byte_array.getvalue()

    return Response(
        [image_byte_array],
        status=200,
        mimetype="image/jpeg")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
