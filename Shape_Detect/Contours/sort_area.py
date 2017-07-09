from __future__ import print_function
import cv2
import numpy as np
import os.path

def get_contour_areas(contours):
    '''Returns the areas of all contours as list'''
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

# Load our image
image = cv2.imread(os.path.dirname(__file__) + '/../../images/bunchofshapes.jpg')
orig = image

# Grayscale our image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 50, 200)

# Find contours and print how many were found
# Python 3: _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # OCV3 probably doesn't need .copy()
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # OCV3 probably doesn't need .copy()

# Let's print the areas of the contours before sorting
print("Contour Areas before sorting")
print(get_contour_areas(contours))

# Sort contours large to small
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
#sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

print("Contour Areas after sorting")
print(get_contour_areas(sorted_contours))

# Iterate over our contours and draw one at a time
# Note that we keep the same window and redraw, rather than opening new windows
for c in sorted_contours:
    cv2.drawContours(orig, [c], -1, (255,0,0), 3)
    cv2.waitKey(0)
    cv2.imshow('Contours by area', orig)

cv2.waitKey(0)
cv2.destroyAllWindows()