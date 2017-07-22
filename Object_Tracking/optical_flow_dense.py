"""
Optical Flow

Requires OpenCV 3.X and FFMPEG 3.X

Seeks to get the pattern of apparent motion of objects in an image between two consecutive frames.
Shows the distribution of the apparent velocities of objects in an image. 

OpenCV has two implementations of Optical Flow.

1. Lucas-Kanade Differential Method - Tracks some keypoints in the video, good for corner-like 
features (tracking cars from drones)
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf
    https://www.cs.cmu.edu/afs/cs/academic/class/15385-s12/www/lec_slides/Baker&Matthews.pdf

2. Dense Optical Flow - Slower, but computes the optical flow for all points in a frame, unlike 
Lucas-Kanade which uses corner features (sparse dataset). Colors are used to reflect movement 
with Hue being direction and value (brightness/intensity) being speed.
"""
import cv2
import numpy as np
import os.path

# Load video stream
source = os.path.dirname(__file__) + '/../images/walking.avi'
cap = cv2.VideoCapture(source)
if cap is None or not cap.isOpened():
    print 'Warning: Unable to open video source: {}'.format(source)

# Get first frame
ret, frame = cap.read()
previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame)
hsv[...,1] = 255

while frame is not None:
    # Read of video file
    ret, frame = cap.read()
    if frame is not None:
        next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # Computes the dense optical flow using the Gunnar Farneback’s algorithm
        flow = cv2.calcOpticalFlowFarneback(previous_gray, next, 
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # use flow to calculate the magnitude (speed) and angle of motion
        # use these values to calculate the color to reflect speed and angle
        magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = angle * (180 / (np.pi/2))
        hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Show our demo of Dense Optical Flow
        cv2.imshow('Dense Optical Flow', final)
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break
        
        # Store current image as previous image
        previous_gray = next

cap.release()
cv2.destroyAllWindows()