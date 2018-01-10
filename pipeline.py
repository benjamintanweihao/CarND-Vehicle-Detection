# Perform a Histogram of Oriented Gradients (HOG) feature extraction on a
# labeled training set of images and train a classifier Linear SVM classifier
# Optionally, you can also apply a color transform and append binned color
# features, as well as histograms of color, to your HOG feature vector.
# Note: for those first two steps don't forget to normalize your features
# and randomize a selection for training and testing.
# Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# Estimate a bounding box for vehicles detected.
import glob
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

car_images = glob.glob('data/vehicles/**/*.png', recursive=True)
non_car_images = glob.glob('data/non-vehicles/*/**.png', recursive=True)


def data_look(cars, non_cars):
    return {'n_cars': len(cars),
            'n_notcars': len(non_cars),
            'image_shape': cv2.imread(cars[0]).shape,
            'data_type': cv2.imread(cars[0]).dtype}


def bin_spatial(image, size=(32, 32)):
    image = np.copy(image)
    small_img = cv2.resize(image, size)
    features = small_img.ravel()

    return features


def color_hist(image, bins=32, bins_range=(0, 256)):
    bhist = np.histogram(image[:, :, 0], bins=bins, range=bins_range)
    ghist = np.histogram(image[:, :, 1], bins=bins, range=bins_range)
    rhist = np.histogram(image[:, :, 2], bins=bins, range=bins_range)
    hist_features = np.concatenate((bhist[0], ghist[0], rhist[0]))

    return hist_features


def get_hog_features(image, orient=9, pix_per_cell=8, cell_per_block=2, feature_vector=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(gray, orientations=orient,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block),
                              visualise=True, feature_vector=feature_vector,
                              block_norm="L2-Hys")

    return features, hog_image


def extract_features(images,
                     spatial_size=(32, 32),
                     hist_bins=32,
                     hist_range=(0, 256)):
    features = []
    for file in tqdm(images):
        image = cv2.imread(file)
        feature_image = np.copy(image)
        spatial_features = bin_spatial(feature_image, spatial_size)
        hist_features = color_hist(feature_image, bins=hist_bins, bins_range=hist_range)

        features.append(np.concatenate((spatial_features, hist_features)))

    return features


def normalize_features(car_features, not_car_features):
    assert len(car_features) > 0
    assert len(non_car_features) > 0

    # create an arary stack of feature vectors
    X = np.vstack((car_features, not_car_features)).astype(np.float64)
    # fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    return scaled_X


car_features = extract_features(car_images[0: 100])
non_car_features = extract_features(non_car_images[0: 100])
normalized_features = normalize_features(car_features, non_car_features)
