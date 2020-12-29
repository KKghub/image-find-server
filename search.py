import os
import constants
import image_retrieval


def results(image):
    result = []

    predicted_class = image_retrieval.predict(image)
    print('predicted_class', predicted_class)
    for filename in os.listdir(os.path.join(os.getcwd(), constants.CLASSIFIED_DATASET_IMAGES_FOLDER, predicted_class)):
        result.append({
            'name': filename,
            'accuracy': '10',
            'class': predicted_class
        })

    return result
