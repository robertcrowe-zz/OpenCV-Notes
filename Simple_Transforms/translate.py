import cv2
import numpy as np
import os.path

image = cv2.imread(os.path.dirname(__file__) + '/../images/input.jpg')
cv2.imshow('Orig', image)
cv2.waitKey()

# Store height and width of the image
height, width = image.shape[:2]

quarter_height, quarter_width = height/4, width/4

#       | 1 0 Tx |
#  T  = | 0 1 Ty |

# T is our translation matrix
T = np.float32([[1, 0, quarter_width], [0, 1,quarter_height]])

# "Affine" - No perspective change or skew - parallel remains parallel
# We use warpAffine to transform the image using the matrix, T
img_translation = cv2.warpAffine(image, T, (width, height))
cv2.imshow('Translation', img_translation)
cv2.waitKey()
cv2.destroyAllWindows()
