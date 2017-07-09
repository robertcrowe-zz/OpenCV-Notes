import cv2
import os.path

# Load our input image
image = cv2.imread(os.path.dirname(__file__) + '/../images/input.jpg')
cv2.imshow('Original', image)
cv2.waitKey()

# We use cvtColor, to convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image)
cv2.waitKey()
cv2.destroyAllWindows()

# Could also do it in the imread() with 0
img = cv2.imread(os.path.dirname(__file__) + '/../images/input.jpg', 0)

cv2.imshow('Grayscale', img)
cv2.waitKey()
cv2.destroyAllWindows()
