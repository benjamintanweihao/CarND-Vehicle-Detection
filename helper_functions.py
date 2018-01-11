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
import time
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm



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
    features, hog_image = hog(image, orientations=orient,
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


def extract_hog_features(images, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in tqdm(images):
        # Read in each one by one
        image = cv2.imread(file)
        feature_image = np.copy(image)
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_feature, _ = get_hog_features(feature_image[:, :, channel],
                                                  orient,
                                                  pix_per_cell,
                                                  cell_per_block)
                hog_features.append(hog_feature)
            hog_features = np.ravel(hog_features)
        else:
            hog_features, _ = get_hog_features(feature_image[:, :, hog_channel],
                                               orient,
                                               pix_per_cell,
                                               cell_per_block)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features


def create_dataset(car_features, not_car_features, test_size=0.2):
    assert len(car_features) > 0
    assert len(non_car_features) > 0

    # create an arary stack of feature vectors
    X = np.vstack((car_features, not_car_features)).astype(np.float64)
    # fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    rand_state = np.random.randint(0, 100)

    return train_test_split(scaled_X, y, test_size=test_size, random_state=rand_state)


#####################
# Drawing Functions #
#####################

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image, start and stop positions in both x and y,
# window size (x and y dimensions), and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list
