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

BRIEF
http://cvlabwww.epfl.ch/~lepetit/papers/calonder_pami11.pdf
"""

image = cv2.imread(os.path.dirname(__file__) + '/../images/input.jpg')
cv2.imshow('Original', image)
cv2.waitKey()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create FAST detector object
fast = cv2.FastFeatureDetector()

# Create BRIEF extractor object
brief = cv2.DescriptorExtractor_create("BRIEF")

# Determine key points
keypoints = fast.detect(gray, None)

# Obtain descriptors and new final keypoints using BRIEF
keypoints, descriptors = brief.compute(gray, keypoints)
print "Number of keypoints Detected: ", len(keypoints)

# Draw rich keypoints on input image
image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                                    
cv2.imshow('Feature Method - BRIEF', image)
cv2.waitKey()
cv2.destroyAllWindows()