import cv2
import numpy as np
import os.path

"""
Speeded Up Robust Features (SURF)

SIFT is quite effective but computationally expensive
SURF was developed to improve the speed of a scale invariant feature 
detector 
Instead of using the Difference of Gaussian approach, SURF uses 
Hessian matrix approximation to detect interesting points and use the 
sum of Haar wavelet responses for orientation assignment.

SURF
http://www.vision.ee.ethz.ch/~surf/eccv06.pdf

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
"""

image = cv2.imread(os.path.dirname(__file__) + '/../images/input.jpg')
cv2.imshow('Original', image)
cv2.waitKey()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create SURF Feature Detector object
surf = cv2.SURF()

# Only features, whose hessian is larger than hessianThreshold are 
# retained by the detector
surf.hessianThreshold = 500
keypoints, descriptors = surf.detectAndCompute(gray, None)
print "Number of keypoints Detected: ", len(keypoints)

# Draw rich key points on input image
image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - SURF', image)
cv2.waitKey()
cv2.destroyAllWindows()
