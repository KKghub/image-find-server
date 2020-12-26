from ImageRetrieval import *

import constants


class Search:

    @staticmethod
    def results():
        result = []

        query_filepath = os.path.join(constants.QUERY_IMAGE)
        # query_image = Image.open(query_filepath)
        # query_feature = ImageProcessor.extract_feature1(query_image)

        predicted_class = ImageRetrieval().predict(query_filepath)

        for filename in os.listdir(os.path.join(constants.CLASSIFIED_DATASET_IMAGES_FOLDER, predicted_class)):
            result.append({
                'name': filename,
                'accuracy': '10',
                'class': predicted_class
            })

        return result
