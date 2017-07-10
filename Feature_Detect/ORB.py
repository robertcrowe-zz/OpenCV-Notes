import cv2
import numpy as np
import os.path

"""
Alternatives to SIFT and SURF ----------------

Features from Accelerated Segment Test (FAST)

  Key point detection only (no descriptor, we can use SIFT or SURF to get that)
  Used in real time applications

Binary Robust Independent Elementary Features (BRIEF)

  Computers descriptors quickly (instead of using SIFT or SURF)
  Fast

Oriented FAST and Rotated BRIEF (ORB) 

  Developed out of OpenCV Labs (not patented so free to use!)
  Combines both Fast and Brief
  http://www.willowgarage.com/sites/default/files/orb_final.pdf

Oriented FAST and Rotated BRIEF (ORB)
http://www.willowgarage.com/sites/default/files/orb_final.pdf
"""

image = cv2.imread(os.path.dirname(__file__) + '/../images/input.jpg')
cv2.imshow('Original', image)
cv2.waitKey()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create ORB object, we can specify the number of key points we desire
orb = cv2.ORB()

# Determine key points
keypoints = orb.detect(gray, None)

# Obtain the descriptors
keypoints, descriptors = orb.compute(gray, keypoints)
print("Number of keypoints Detected: ", len(keypoints))

# Draw rich keypoints on input image
image = cv2.drawKeypoints(image, keypoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - ORB', image)
cv2.waitKey()
cv2.destroyAllWindows()