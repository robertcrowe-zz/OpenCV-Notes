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

import cv2
import numpy as np

# Initalize webacam and store first frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Create a flaot numpy array with frame values
average = np.float32(frame)

while True:
    # Get webcam frmae
    ret, frame = cap.read()
    
    if frame is not None:
        # 0.01 is the weight of image, play around to see how it changes
        cv2.accumulateWeighted(frame, average, 0.01)
        
        # Scales, calculates absolute values, and converts the result to 8-bit
        background = cv2.convertScaleAbs(average)

        cv2.imshow('Input', frame)
        cv2.imshow('Ghosting Foreground', background)
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cv2.destroyAllWindows()
cap.release()
