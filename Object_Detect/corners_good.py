import cv2
import numpy as np
import os.path

"""
Problems with corners as features

Corner matching in images is tolerant of:
Rotations
Translations (i.e. shifts in image)
Slight photometric changes e.g. brightness or affine intensity

However, it is intolerant of:
Large changes in intensity or photometric changes)
Scaling (i.e. enlarging or shrinking)

Good Features is an improvement of Harris

cv2.goodFeaturesToTrack(input image, maxCorners, qualityLevel, minDistance)

Input Image - 8-bit or floating-point 32-bit, single-channel image.
maxCorners – Maximum number of corners to return. If there are more 
corners than are found, the strongest of them is returned.

qualityLevel – Parameter characterizing the minimal accepted quality 
of image corners. The parameter value is multiplied by the best corner 
quality measure (smallest eigenvalue). The corners with the quality 
measure less than the product are rejected. For example, if the best 
corner has the quality measure = 1500, and the qualityLevel=0.01 , then 
all the corners with the quality measure less than 15 are rejected.
Higher means higher quality match.

minDistance – Minimum possible Euclidean distance between the returned 
corners.
"""

img = cv2.imread(os.path.dirname(__file__) + '/../images/chess.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# We specific the top 50 corners
corners = cv2.goodFeaturesToTrack(gray, 50, 0.1, 50)

for corner in corners:
    x, y = corner[0]
    x = int(x)
    y = int(y)
    cv2.rectangle(img, (x-10,y-10), (x+10,y+10), (0,255,0), 2)
    
cv2.imshow("Corners Found", img)
cv2.waitKey()
cv2.destroyAllWindows()
