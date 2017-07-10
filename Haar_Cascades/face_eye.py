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

dirname = os.path.dirname(__file__)
face_classifier = cv2.CascadeClassifier(dirname + '/../Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(dirname + '/../Haarcascades/haarcascade_eye.xml')
 
img = cv2.imread(dirname + '/../images/Trump.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# When no faces detected, face_classifier returns and empty tuple
if faces is ():
    print("No Face Found")

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    
cv2.destroyAllWindows()