# Standard imports
import cv2
import numpy as np
import os.path

"""
cv::SimpleBlobDetector: Class for extracting blobs from an image

The class implements a simple algorithm for extracting blobs from an image:

Convert the source image to binary images by applying thresholding with several 
thresholds from minThreshold (inclusive) to maxThreshold (exclusive) with distance 
thresholdStep between neighboring thresholds.
Extract connected components from every binary image by findContours and calculate 
their centers.
Group centers from several binary images by their coordinates. Close centers form 
one group that corresponds to one blob, which is controlled by the minDistBetweenBlobs 
parameter.
From the groups, estimate final centers of blobs and their radiuses and return as 
locations and sizes of keypoints.
This class performs several filtrations of returned blobs. You should set filterBy* 
to true/false to turn on/off corresponding filtration. Available filtrations:

By color. This filter compares the intensity of a binary image at the center of a blob 
to blobColor. If they differ, the blob is filtered out. Use blobColor = 0 to extract 
dark blobs and blobColor = 255 to extract light blobs.
By area. Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
By circularity. Extracted blobs have circularity ( 4∗π∗Areaperimeter∗perimeter) between 
minCircularity (inclusive) and maxCircularity (exclusive).
By ratio of the minimum inertia to maximum inertia. Extracted blobs have this ratio 
between minInertiaRatio (inclusive) and maxInertiaRatio (exclusive).
By convexity. Extracted blobs have convexity (area / area of blob convex hull) between 
minConvexity (inclusive) and maxConvexity (exclusive).
Default values of parameters are tuned to extract dark circular blobs.

The function cv2.drawKeypoints takes the following arguments:
cv2.drawKeypoints(input image, keypoints, blank_output_array, color, flags)

flags:
cv2.DRAW_MATCHES_FLAGS_DEFAULT
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
"""
# Read image
image = cv2.imread(os.path.dirname(__file__) + '/../images/Sunflowers.jpg')
 
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector()
 
# Detect blobs.
keypoints = detector.detect(image)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of
# the circle corresponds to the size of blob
blank = np.zeros((1,1)) 
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,255,255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Blobs", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
