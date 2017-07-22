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
import numpy as np
import cv2
import os.path

# Load video stream
source = os.path.dirname(__file__) + '/../images/walking.avi'
cap = cv2.VideoCapture(source)
if cap is None or not cap.isOpened():
    print 'Warning: Unable to open video source: {}'.format(source)

# Set parameters for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Set parameters for lucas kanade optical flow
lucas_kanade_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
# Used to create our trails for object movement in the image 
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, frame = cap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Find inital corner locations
prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(frame)

while frame is not None:
    ret, frame = cap.read()
    if frame is not None:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray, 
                                                            frame_gray, 
                                                            prev_corners, 
                                                            None, 
                                                            **lucas_kanade_params)

        # Select and store good points
        good_new = new_corners[status==1]
        good_old = prev_corners[status==1]

        # Draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a,b), 5, color[i].tolist(),-1)
            
        img = cv2.add(frame,mask)

        # Show Optical Flow
        cv2.imshow('Optical Flow - Lucas-Kanade',img)
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

        # Now update the previous frame and previous points
        prev_gray = frame_gray.copy()
        prev_corners = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
