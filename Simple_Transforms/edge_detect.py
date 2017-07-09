import cv2
import numpy as np
import os.path

image = cv2.imread(os.path.dirname(__file__) + '/../images/elephant.jpg',0)
cv2.imshow('Original', image)
cv2.waitKey(0)

height, width = image.shape

# Extract Sobel Edges
sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

cv2.imshow('Sobel X', sobel_x)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobel_y)
cv2.waitKey(0)

sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
cv2.imshow('sobel_OR', sobel_OR)
cv2.waitKey(0)

laplacian = cv2.Laplacian(image, cv2.CV_64F)
cv2.imshow('Laplacian', laplacian)
cv2.waitKey(0)

"""
Then, we need to provide two values: threshold1 and threshold2. Any gradient value larger than threshold2
is considered to be an edge. Any value below threshold1 is considered not to be an edge. 
Values in between threshold1 and threshold2 are either classiﬁed as edges or non-edges based on how their 
intensities are "connected". In this case, any gradient values below 50 are considered non-edges
whereas any values above 120 are considered edges.
"""

# Canny Edge Detection uses gradient values as thresholds
# The first threshold gradient
canny = cv2.Canny(image, 50, 120)
cv2.imshow('Canny', canny)
cv2.waitKey(0)

cv2.destroyAllWindows()