import numpy as np
import cv2
import os.path

"""
What are HAAR Cascade Classifiers?

An object detection method that inputs Haar features into a series of classifiers 
(cascade) to identify objects in an image.  They are trained to identify objects in an image. 
They are trained to identify one type of object, however, we can use several of them in parallel,
for example detecting eyes and faces together.

Other pre-trained classifiers:
https://github.com/opencv/opencv/tree/master/data/haarcascades

Tuning Cascade Classifiers

ourClassifier.detectMultiScale(input image, Scale Factor , Min Neighbors)

Scale Factor Specifies how much we reduce the image size each time we scale. E.g. in face 
detection we typically use 1.3. This means we reduce the image by 30% each time it’s scaled. 
Smaller values, like 1.05 will take longer to compute, but will increase the rate of detection.

Min Neighbors Specifies the number of neighbors each potential window should have in order to 
consider it a positive detection. Typically set between 3-6. It acts as sensitivity setting, 
low values will sometimes detect multiples faces over a single face. High values will ensure 
less false positives, but you may miss some faces.
"""

# We point OpenCV's CascadeClassifier function to where our 
# classifier (XML file format) is stored
face_classifier = cv2.CascadeClassifier(os.path.dirname(__file__) + '/../Haarcascades/haarcascade_frontalface_default.xml')

# Load our image then convert it to grayscale
image = cv2.imread(os.path.dirname(__file__) + '/../images/Trump.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Our classifier returns the ROI of the detected face as a tuple
# It stores the top left coordinate and the bottom right coordiantes
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# When no faces detected, face_classifier returns and empty tuple
if faces is ():
    print("No faces found")

# We iterate through our faces array and draw a rectangle
# over each face in faces
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()