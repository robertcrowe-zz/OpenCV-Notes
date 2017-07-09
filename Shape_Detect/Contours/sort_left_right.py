import cv2
import numpy as np
import os.path

def x_cord_contour(contours):
    '''Returns the X cordinate for the contour centroid'''
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))

def label_contour_center(image, c):
    '''Places a red circle on the centers of contours'''
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
 
    # Draw the countour number on the image
    cv2.circle(image,(cx,cy), 10, (0,0,255), -1)
    return image

# Load our image
image = cv2.imread(os.path.dirname(__file__) + '/../../images/bunchofshapes.jpg')
orig = image.copy()

# Grayscale our image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 50, 200)

# Find contours and print how many were found
# Python 3: _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # OCV3 probably doesn't need .copy()
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # OCV3 probably doesn't need .copy()

# Computer Center of Mass or centroids and draw them on our image
for (i, c) in enumerate(contours):
    orig = label_contour_center(image, c)
 
cv2.imshow("4 - Contour Centers ", image)
cv2.waitKey(0)

# Sort by left to right using our x_cord_contour function
contours_left_to_right = sorted(contours, key=x_cord_contour, reverse=False)

# Labeling Contours left to right
for (i,c)  in enumerate(contours_left_to_right):
    cv2.drawContours(orig, [c], -1, (0,0,255), 3)  
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(orig, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('6 - Left to Right Contour', orig)
    cv2.waitKey(0)
    (x, y, w, h) = cv2.boundingRect(c)  
    
    # Let's now crop each contour and save these images to files
    # cropped_contour = orig[y:y + h, x:x + w]
    # image_name = "output_shape_number_" + str(i+1) + ".jpg"
    # print(image_name)
    # cv2.imwrite(image_name, cropped_contour)
    
cv2.destroyAllWindows()


