import cv2
import numpy as np
import os.path

image = cv2.imread(os.path.dirname(__file__) + '/../images/input.jpg')

# Create a matrix of ones, then multiply it by a scaler of 100 
# This gives a matrix with same dimensions of our image with all values being 100
M = np.ones(image.shape, dtype="uint8") * 100 

# We use this to add this matrix M, to our image
# Notice the increase in brightness
added = cv2.add(image, M)
cv2.imshow("Added - Brightness", added)
cv2.waitKey(0)

# Likewise we can also subtract
# Notice the decrease in brightness
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted - Brightness", subtracted)
cv2.waitKey(0)
cv2.destroyAllWindows()