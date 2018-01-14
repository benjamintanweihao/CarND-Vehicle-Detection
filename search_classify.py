import glob
import time

import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage import label
from sklearn.svm import LinearSVC
from tqdm import tqdm

from helper_functions import *
import matplotlib.image as mpimg

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())


car_images = glob.glob('data/vehicles/**/*.png', recursive=True)
non_car_images = glob.glob('data/non-vehicles/*/**.png', recursive=True)

# sample_size = 1000
# cars = car_images[0:sample_size]
# non_cars = non_car_images[0:sample_size]

cars = car_images
non_cars = non_car_images

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [None, None]  # Min and max in y to search in slide_window()

car_features = []
for file in tqdm(cars):
    img = mpimg.imread(file)

    car_features.append(single_img_features(img,
                                            color_space='YCrCb',
                                            spatial_size=spatial_size,
                                            hist_bins=hist_bins,
                                            orient=orient,
                                            pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block,
                                            hog_channel=hog_channel,
                                            spatial_feat=spatial_feat,
                                            hist_feat=hist_feat,
                                            hog_feat=hog_feat))

non_car_features = []
for file in tqdm(non_cars):
    img = mpimg.imread(file)
    non_car_features.append(single_img_features(img,
                                                color_space='YCrCb',
                                                spatial_size=spatial_size,
                                                hist_bins=hist_bins,
                                                orient=orient,
                                                pix_per_cell=pix_per_cell,
                                                cell_per_block=cell_per_block,
                                                hog_channel=hog_channel,
                                                spatial_feat=spatial_feat,
                                                hist_feat=hist_feat,
                                                hog_feat=hog_feat))

result = create_dataset(car_features, non_car_features)

X_train = result['X_train']
X_test = result['X_test']
y_train = result['y_train']
y_test = result['y_test']
X_scaler = result['X_scaler']

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()


def pipeline(img):
    rectangles = []

    for ystart, ystop, scale in [[400, 656, 1.5],
                                 [400, 464, 1.0],
                                 [416, 480, 1.0],
                                 [400, 496, 1.5],
                                 [432, 528, 1.5],
                                 [400, 528, 2.0],
                                 [432, 560, 2.0],
                                 [400, 596, 3.5],
                                 [464, 660, 3.5]]:

        out_img, heat_map, rects = find_cars(img, scale=scale,
                                             ystart=ystart,
                                             ystop=ystop,
                                             pix_per_cell=pix_per_cell,
                                             cell_per_block=cell_per_block,
                                             orient=orient,
                                             spatial_size=spatial_size,
                                             hist_bins=hist_bins,
                                             X_scaler=X_scaler,
                                             svc=svc)

        rectangles.append(rects)

    print(rectangles)

    rectangles = [item for sublist in rectangles for item in sublist]

    heatmap_img = np.zeros_like(img[:, :, 0])
    heatmap_img = add_heat(heatmap_img, rectangles)
    heatmap_img = apply_threshold(heatmap_img, 1)
    labels = label(heatmap_img)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img


video_file_name = "project_video.mp4"
write_output = 'output_video/' + video_file_name
clip1 = VideoFileClip(video_file_name)
clip2 = clip1.fl_image(pipeline)
clip2.write_videofile(write_output, audio=False)
