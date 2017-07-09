import cv2
import os.path

# An image pyramid is a collection of images - all arising from 
# a single original image - that are successively downsampled 
# until some desired stopping point is reached.
# There are two common kinds of image pyramids:
# Gaussian pyramid: Used to downsample images
# Laplacian pyramid: Used to reconstruct an upsampled image from 
# an image lower in the pyramid (with less resolution)

image = cv2.imread(os.path.dirname(__file__) + '/../images/input.jpg')

smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(smaller)

cv2.imshow('Original', image )
cv2.waitKey(0)
cv2.imshow('Smaller ', smaller )
cv2.waitKey(0)
cv2.imshow('Larger ', larger )
cv2.waitKey(0)
cv2.destroyAllWindows()
