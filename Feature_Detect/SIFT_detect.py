import cv2
import numpy as np
import os.path

"""
Flann based matching is quite fast, but not the most accurate. Other matching methods include:

BruteForce
BruteForce-SL2 (not in the documentation, BUT this is the one that keeps the squared root!)
BruteForce-L1
BruteForce-Hamming
BruteForce-Hamming(2)

http://docs.opencv.org/trunk/dc/dc3/tutorial_py_matcher.html
"""

def sift_detector(new_image, image_template):
    """Compares input image to template
    Returns (int): the number of SIFT matches between them
    """
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    image2 = image_template
    
    # Create SIFT detector object
    sift = cv2.SIFT()

    # Obtain the keypoints and descriptors using SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    # Define parameters for our Flann Matcher
    FLANN_INDEX_KDTREE = 0
    index_params = {'algorithm':FLANN_INDEX_KDTREE, 'trees':3}
    search_params = {'checks':100}

    # Create the Flann Matcher object
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Obtain matches using K-Nearest Neighbor Method
    # the result is the number of similar matches found in both images
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Store good matches using Lowe's ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m) 

    return len(good_matches)


cap = cv2.VideoCapture(0) # get the webcam

# Load our image template, this is our reference image
image_template = cv2.imread(os.path.dirname(__file__) + '/../images/box_in_scene.png', 0)
cv2.imshow('Template', image_template)
cv2.waitKey(0)

# Get first webcam image
ret, frame = cap.read()

# Get height and width of webcam frame
height, width = frame.shape[:2]

# Define ROI Box Dimensions
top_left_x = width / 3
top_left_y = (height / 2) + (height / 4)
bottom_right_x = (width / 3) * 2
bottom_right_y = (height / 2) - (height / 4)

while True:
    # Get webcam images
    ret, frame = cap.read()
    
    # Draw rectangular window for our region of interest   
    cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), 255, 3)
    
    # Crop region of interest we defined above
    cropped = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]
    
    # Flip frame orientation horizontally for convenience
    frame = cv2.flip(frame,1)
    
    # Get number of SIFT matches
    matches = sift_detector(cropped, image_template)

    # Display status string showing the current no. of matches 
    cv2.putText(frame,str(matches),(450,450), cv2.FONT_HERSHEY_COMPLEX, 2,(0,255,0),1)
    
    # Our threshold to indicate object deteciton
    # We use 10 since the SIFT detector returns little false positves
    threshold = 10
    
    # If matches exceed our threshold then object has been detected
    if matches > threshold:
        cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (0,255,0), 3)
        cv2.putText(frame,'Object Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)
    
    cv2.imshow('Object Detector using SIFT', frame)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()   