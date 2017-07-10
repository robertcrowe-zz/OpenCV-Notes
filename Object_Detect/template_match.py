import cv2
import numpy as np
import os.path

"""
Notes on Template Matching

There are a variety of methods to perform template matching, but in this 
case we are using the correlation coefficient which is specified by the 
flag cv2.TM_CCOEFF.

So what exactly is the cv2.matchTemplate function doing? Essentially, this 
function takes a sliding window of our waldo query image and slides it 
across our puzzle image from left to right and top to bottom, one pixel at 
a time. Then, for each of these locations, we compute the correlation 
coefficient to determine how good or bad the match is.

Regions with sufficiently high correlation can be considered matches for 
our waldo template. From there, all we need is a call to cv2.minMaxLoc
to find where our good matches are.

http://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html
"""

# Load input image and convert to grayscale
dirname = os.path.dirname(__file__) + '/../images/'
image = cv2.imread(dirname + 'WaldoBeach.jpg')
cv2.imshow('Where is Waldo?', image)
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Template image - note grayscale (0)
template = cv2.imread(dirname + 'waldo.jpg', 0)

result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Create Bounding Box
top_left = max_loc
bottom_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(image, top_left, bottom_right, (0,0,255), 5)

cv2.imshow('Where is Waldo?', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
