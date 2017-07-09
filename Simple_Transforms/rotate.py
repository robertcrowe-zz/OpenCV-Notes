import cv2
import numpy as np
import os.path

image = cv2.imread(os.path.dirname(__file__) + '/../images/input.jpg')
height, width = image.shape[:2]

# Divide by two to rotate the image around its center (counterclockwise) ---------
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, .5)

rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()

# Another option to rotate 90 --------------
rotated_image = cv2.transpose(image)

cv2.imshow('Rotated Image - Method 2', rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()

# Let's now to a horizontal flip.
flipped = cv2.flip(image, 1)
cv2.imshow('Horizontal Flip', flipped) 
cv2.waitKey()
cv2.destroyAllWindows()