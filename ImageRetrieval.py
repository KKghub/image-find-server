import cv2
# import math
# import re
import constants
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
# from skimage import data, io, exposure
# from skimage.filters import threshold_otsu
# from skimage.measure import label
# from skimage.morphology import closing, square, opening, disk
# from skimage.measure import regionprops, perimeter
# from skimage.color import label2rgb, rgb2lab, rgb2hsv
# from skimage.util.shape import view_as_blocks
# from skimage.transform import resize
import joblib
# from skimage.segmentation import clear_border
from skimage.feature import greycomatrix, greycoprops
import os


# def get_hog_descriptor(image):
# 	fd,_ = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)
# 	return fd

class ImageRetrieval:

    def build_feature_file(self):
        # '''
        # takes a directory with images and builds a file with feature descriptors and labels
        # '''

        image_urls = []
        data_dir = os.path.join(os.getcwd(), constants.CLASSIFIED_DATASET_IMAGES_FOLDER)
        mapper = {}
        label = 0
        with open('datafile.csv', 'a') as data_file:
            for category in os.listdir(data_dir):
                # if f not in ground_truth:
                # continue
                label += 1
                print("***** ", category)
                mapper[label] = category
                img_dir = os.path.join(data_dir, category)
                for img_path in os.listdir(img_dir):
                    try:
                        print(img_path)
                        # try:
                        fd = ','.join([str(i) for i in self.get_image_features(os.path.join(img_dir, img_path))])
                        fd += ',' + str(label) + '\n'
                        data_file.write(fd)

                    except Exception as e:
                        print(e)

                        pass

        joblib.dump(mapper, 'mapper.pkl')

    def save_model(self, clf):
        joblib.dump(clf, 'model.pkl')

    def learn(self):
        from sklearn import svm
        d = np.genfromtxt('datafile.csv', delimiter=',')
        print(d.shape)
        X = d[:, :d.shape[1] - 1]
        y = d[:, d.shape[1] - 1]

        pca = PCA(.95)
        X = pca.fit_transform(X)

        clf = svm.SVC()
        clf.fit(X, y)
        # save model
        self.save_model(clf)
        joblib.dump(pca, "pca.pkl")

    def predict(self, image_path):
        print(os.getcwd())
        clf = joblib.load('model.pkl')
        mapper = joblib.load('mapper.pkl')
        pca = joblib.load("pca.pkl")

        print(mapper)
        a = self.get_image_features(image_path)
        a = a.reshape(1, -1)
        a = pca.transform(a)
        print(mapper.get(int(clf.predict(a)[0])))
        return mapper.get(int(clf.predict(a)[0]))

    def predict_by_image(self, image):
        print(os.getcwd())
        clf = joblib.load('model.pkl')
        mapper = joblib.load('mapper.pkl')
        pca = joblib.load("pca.pkl")

        print(mapper)
        print('1')
        a = self.get_image_features_by_image(image)
        print('2')
        a = a.reshape(1, -1)
        a = pca.transform(a)
        print(clf.predict(a)[0])
        print(mapper.get(int(clf.predict(a)[0])))
        return mapper.get(int(clf.predict(a)[0]))

    # return mapper.get()

    @staticmethod
    def extract_features(regionprops, image):

        region_count = len(regionprops)

        average_area = np.average([region.filled_area for region in regionprops])
        max_area = max(regionprops, key=lambda region: region.filled_area).filled_area
        average_perimeter = np.average([region.perimeter for region in regionprops])
        average_euler_number = np.average([region.euler_number for region in regionprops])
        average_eccentricity = np.average([region.eccentricity for region in regionprops])
        average_equivalent_diameter = np.average([region.equivalent_diameter for region in regionprops])
        mean_intensity = np.average([region.mean_intensity for region in regionprops])
        glcm = greycomatrix(image, [1, 2], [0, np.pi / 2], levels=256, symmetric=True, normed=True)
        dissimilarity = np.mean(greycoprops(glcm, 'dissimilarity'))
        correlation = np.mean(greycoprops(glcm, 'correlation'))
        energy = np.mean(greycoprops(glcm, 'energy'))
        homogeneity = np.mean(greycoprops(glcm, 'homogeneity'))
        ASM = np.mean(greycoprops(glcm, 'ASM'))
        contrast = np.mean(greycoprops(glcm, 'contrast'))

        return [region_count, average_eccentricity, average_perimeter, max_area, average_area,
                average_equivalent_diameter,
                average_euler_number, mean_intensity, dissimilarity, correlation, energy, homogeneity, ASM, contrast]

    def get_image_features(self, image_path):
        bins = 8
        image = cv2.imread(image_path)
        image = cv2.resize(image, (200, 200))
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])  # flating
        cv2.normalize(hist, hist)
        features = hist.flatten()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        std_image = np.float32(gray_image) / 255.0
        dct = cv2.dct(std_image)

        features = np.concatenate((features, dct), axis=None)

        # print(final_features.shape)
        return features

    def get_image_features_by_image(self, image):
        bins = 8
        image = cv2.resize(image, (200, 200))
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])  # flating
        cv2.normalize(hist, hist)
        features = hist.flatten()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        std_image = np.float32(gray_image) / 255.0
        dct = cv2.dct(std_image)

        features = np.concatenate((features, dct), axis=None)

        # print(final_features.shape)
        return features

# if __name__ == '__main__':
# 	build_feature_file()
# 	learn()
# print(predict('test_young.jpg'))
# predict('kane.jpg')
