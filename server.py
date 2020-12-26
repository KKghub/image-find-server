import io
import json
from flask_cors import CORS
from flask import Flask, request, Response
from werkzeug.datastructures import FileStorage
from PIL import Image
from utilities import *

app = Flask(__name__)
CORS(app)


@app.route('/queryImage', methods=['POST'])
def post_query_image():
    image = request.files.get('image')  # type: FileStorage
    # print(request.files)
    image.save(os.path.join(constants.QUERY_IMAGE))
    # return Response(status=204)
    return Response("{'a':'b'}", status=200, mimetype='application/json')


@app.route('/getResults', methods=['GET'])
def get_results():
    results = Search().results()

    payload = json.dumps(results).encode()

    return Response(
        payload,
        status=200,
        mimetype="application/json"
        # headers={
        #     "Content-Disposition": "attachment;filename=".join(filenamepath)
        # }
    )


@app.route('/getImage', methods=['GET'])
def get_image():
    print("1")
    filename = request.args.get("name")
    filenamepath = os.path.join(constants.DATASET_IMAGES_FOLDER, filename)

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
    print("1")
    filename = request.args.get("name")
    filenamepath = os.path.join(constants.CLASSIFIED_DATASET_IMAGES_FOLDER, class_name, filename)

    print(os.path.exists(filenamepath))
    print(filenamepath)
    image = Image.open(filenamepath)
    print("2")
    image_byte_array = io.BytesIO()
    print("3")
    image.save(image_byte_array, format=image.format)
    print("4")
    image_byte_array = image_byte_array.getvalue()

    return Response(
        [image_byte_array],
        status=200,
        mimetype="image/jpeg")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
