"""
Filter by Color

Basically, this is how a green screen works.  It creates a mask
using color filtering and then ands the mask with the original
image.

Implementing a Color Filter

1.Define upper and lower range of color filter.
2.Create a binary (thresholded) mask showing only the desired colors 
in white:

    mask= cv2.inRange(hsv_img, lower_color_range, upper_color_range)

3.Perform a bitwise_and operation on the original image and the mask.
"""

import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# define range of purple color in HSV
lower_purple = np.array([125,0,0])
upper_purple = np.array([175,255,255])

# loop and wait for enter key
while True:
    # Read webcam image
    ret, frame = cap.read()
    
    # Convert image from RBG/BGR to HSV so we easily filter
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Use inRange to capture only the values between lower & upper purple
    mask = cv2.inRange(hsv_img, lower_purple, upper_purple)

    # Perform Bitwise AND on mask and our original frame
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame)  
    cv2.imshow('mask', mask)
    cv2.imshow('Filtered Color Only', res)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()