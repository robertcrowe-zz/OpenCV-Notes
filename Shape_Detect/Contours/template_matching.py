import cv2
import numpy as np
import os.path

"""
cv2.matchShapes(contour template, contour, method, method parameter)

Output – match value (lower values means a closer match)
Contour Template – This is our reference contour that we’re trying to find in the new image
Contour – The individual contour we are checking against
Method – Type of contour matching (1, 2, 3)
Method Parameter – leave alone as 0.0 (not fully utilized in python OpenCV)

http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
"""

# Load the shape template or reference image
template = cv2.imread(os.path.dirname(__file__) + '/../images/4star.jpg',0)
cv2.imshow('Template', template)
cv2.waitKey()

# Load the target image with the shapes we're trying to match
target = cv2.imread(os.path.dirname(__file__) + '/../images/shapestomatch.jpg')
target_gray = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)

# Threshold both images first before using cv2.findContours
ret, thresh1 = cv2.threshold(template, 127, 255, 0)
ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)

# Find contours in template
_, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# We need to sort the contours by area so that we can remove the largest
# contour which is the image outline (frame)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# We extract the second largest contour which will be our template contour
template_contour = contours[1]

# Extract contours from the target image
_, contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

best_match = (1.0, [])
for c in contours:
    # Iterate through each contour in the target image and 
    # use cv2.matchShapes to compare contour shapes
    # Lower values indicate a better match
    match = cv2.matchShapes(template_contour, c, 3, 0.0)
    print(match)
    if match <= best_match[0]:
        best_match = (match, c)
                
cv2.drawContours(target, [best_match[1]], -1, (0,255,0), 3)
cv2.imshow('Output', target)
cv2.waitKey()
cv2.destroyAllWindows()