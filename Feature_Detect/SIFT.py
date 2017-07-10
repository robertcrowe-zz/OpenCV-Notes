import cv2
import numpy as np
import os.path

"""
SIFT - Scale Invariant Feature Transform
Needed tolerance to scaling (known as scale invariance)
SIFT is widely used (although patented) in computer vision as it 
very successfully dealt with the scale invariance issue
Patented and no longer freely available with OpenCV 3.0+

The SIFT & SURF algorithms are patented by their respective creators, 
and while they are free to use in academic and research settings, 
you should technically be obtaining a license/permission from the 
creators if you are using them in a commercial (i.e. for-profit) 
application.

SIFT
http://www.inf.fu-berlin.de/lehre/SS09/CV/uebungen/uebung09/SIFT.pdf

SIFT in a nutshell

1. Detect interesting key points in an image using the Difference of 
Gaussian method. These are areas of the image where variation exceeds 
a certain threshold and are better than edge descriptors.

2. Create vector descriptors for these interesting areas. Scale 
invariance is achieved via the following process:
  i. Interest points are scanned at several different scales
  ii. The scale at which we meet a specific stability criteria, is 
  then selected and is encoded by the vector descriptor. Therefore, 
  regardless of the initial size, the more stable scale is found which 
  allows us to be scale invariant.

3. Rotation invariance is achieved by obtaining the Orientation Assignment 
of the key point using image gradient magnitudes. Once we know the 2D 
direction, we can normalize this direction.

4.The full paper on SIFT can be read here:
http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

5.An excellent tutorial on SIFT also available here
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html

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
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create SIFT Feature Detector object
sift = cv2.SIFT()

# Detect key points
keypoints = sift.detect(gray, None)
print("Number of keypoints Detected: ", len(keypoints))

# Draw rich key points on input image
image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - SIFT', image)
cv2.waitKey(0)
cv2.destroyAllWindows()