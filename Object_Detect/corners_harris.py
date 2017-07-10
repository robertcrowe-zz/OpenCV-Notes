import cv2
import numpy as np
import os.path

"""
Problems with corners as features

Corner matching in images is tolerant of:
Rotations
Translations (i.e. shifts in image)
Slight photometric changes e.g. brightness or affine intensity

However, it is intolerant of:
Large changes in intensity or photometric changes)
Scaling (i.e. enlarging or shrinking)

Harris Corner Detection is an algorithm developed in 1998 for corner 
detection (http://www.bmva.org/bmvc/1988/avc-88-023.pdf) and works 
fairly well.

cv2.cornerHarris(input image, block size, ksize, k)

Input image - should be grayscale and float32 type.
blockSize - the size of neighborhood considered for corner detection
ksize - aperture parameter of Sobel derivative used.
k - harris detector free parameter in the equation
Output – array of corner locations (x,y)
"""

# Load image then grayscale
image = cv2.imread(os.path.dirname(__file__) + '/../images/chess.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# The cornerHarris function requires the array datatype to be float32
gray = np.float32(gray)

harris_corners = cv2.cornerHarris(gray, 3, 3, 0.05)

# We use dilation of the corner points to enlarge them
kernel = np.ones((7,7),np.uint8)
harris_corners = cv2.dilate(harris_corners, kernel, iterations = 2)

# Threshold for an optimal value, it may vary depending on the image.
image[harris_corners > 0.025 * harris_corners.max() ] = [255, 127, 127]

cv2.imshow('Harris Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()