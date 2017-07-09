from __future__ import print_function
import cv2
import numpy as np
import os.path # Need this to read from parent directory

print('Using OpenCV {}'.format(cv2.__version__))

# Load an image using 'imread' specifying the path to image
input = cv2.imread(os.path.dirname(__file__) + '/../images/input.jpg')
cv2.imshow('Hello World', input)
cv2.waitKey()
cv2.destroyAllWindows()

print(input.shape)
print('Height of Image:', int(input.shape[0]), 'pixels')
print('Width of Image: ', int(input.shape[1]), 'pixels')

# Simply use 'imwrite' specificing the file name and the image to be saved
# cv2.imwrite('output.jpg', input) # file extension determines format
# cv2.imwrite('output.png', input)