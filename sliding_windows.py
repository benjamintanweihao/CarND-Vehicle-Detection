from helper_functions import slide_window, draw_boxes

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('test_images/test1.jpg')

windows = slide_window(image, x_start_stop=[None, None],
                       y_start_stop=[None, None],
                       xy_window=(64, 64), xy_overlap=(0.5, 0.5))

window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
plt.imshow(window_img)
plt.show()

print(image.shape)
