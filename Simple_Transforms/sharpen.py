import cv2
import numpy as np
import os.path

image = cv2.imread(os.path.dirname(__file__) + '/../images/input.jpg')
cv2.imshow('Original', image)

# Create our shapening kernel, we don't normalize since the 
# the values in the matrix sum to 1
# kernel_sharpening = np.array([[-1,-1,-1], 
#                               [-1,9,-1], 
#                               [-1,-1,-1]])
kernel_sharpening = np.array([[-0.1,-0.1,-0.1], 
                              [-0.1,1.8,-0.1], 
                              [-0.1,-0.1,-0.1]])

# applying different kernels to the input image
sharpened = cv2.filter2D(image, -1, kernel_sharpening)

cv2.imshow('Image Sharpening', sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()
