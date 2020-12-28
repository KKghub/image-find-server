from ImageRetrieval import *


class Search:

    @staticmethod
    def results():
        result = []

        query_filepath = os.path.join(os.getcwd(), constants.QUERY_IMAGE_FOLDER, constants.QUERY_IMAGE)
        # query_image = Image.open(query_filepath)
        # query_feature = ImageProcessor.extract_feature1(query_image)

        predicted_class = ImageRetrieval().predict(query_filepath)

        for filename in os.listdir(os.path.join(os.getcwd(), constants.CLASSIFIED_DATASET_IMAGES_FOLDER, predicted_class)):
            result.append({
                'name': filename,
                'accuracy': '10',
                'class': predicted_class
            })

        return result

    @staticmethod
    def results_by_image(image):
        result = []

        predicted_class = ImageRetrieval().predict_by_image(image)
        print('predicted_class', predicted_class)
        for filename in os.listdir(os.path.join(os.getcwd(), constants.CLASSIFIED_DATASET_IMAGES_FOLDER, predicted_class)):
            result.append({
                'name': filename,
                'accuracy': '10',
                'class': predicted_class
            })

        return result
