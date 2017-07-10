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

FAST
https://www.edwardrosten.com/work/rosten_2006_machine.pdf
http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV1011/AV1FeaturefromAcceleratedSegmentTest.pdf
"""
image = cv2.imread(os.path.dirname(__file__) + '/../images/input.jpg')
cv2.imshow('Original', image)
cv2.waitKey()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create FAST Detector object
fast = cv2.FastFeatureDetector()

# Obtain Key points, by default non max suppression is On
# to turn off set fast.setBool('nonmaxSuppression', False)
keypoints = fast.detect(gray, None)
print "Number of keypoints Detected: ", len(keypoints)

# Draw rich keypoints on input image
image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - FAST', image)
cv2.waitKey()
cv2.destroyAllWindows()