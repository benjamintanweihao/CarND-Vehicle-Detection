import cv2
from skimage.feature import hog

from helper_functions import get_hog_features

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('data/vehicles/KITTI_extracted/4.png')
feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

_, hog_image = hog(feature_image[:, :, 2], orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          transform_sqrt=False,
                          visualise=True,
                          feature_vector=False)

plt.imshow(hog_image)
plt.show()


