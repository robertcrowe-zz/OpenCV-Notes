
"""
Background Subtraction

This is a very useful computer vision technique which allows 
us to separate foregrounds from the backgrounds in a video stream.

These algorithms essentially learn about the frame in view 
(video stream) and are able to accurate learn and identify 
the foreground mask. What results is a binary segmentation of the 
image which highlights regions of non-stationary objects.

There are a several Background subtraction algorithms in OpenCV 
specifically for video analysis:

BackgroundSubtractorMOG - Gaussian Mixture-based background/foreground 
Segmentation Algorithm.

BackgroundSubtractorMOG2 – Another Gaussian Mixture-based background 
subtraction method, however with better adaptability to illumination 
changes and with better ability to detect shadows!

Geometric Multigrid (GMG) -This method combines statistical background 
image estimation and per-pixel Bayesian segmentation
"""
# OpenCV 3.1.0
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorKNN()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame',fgmask)
    
    if cv2.waitKey(1) == 13: 
        break

cap.release()
cv2.destroyAllWindows()