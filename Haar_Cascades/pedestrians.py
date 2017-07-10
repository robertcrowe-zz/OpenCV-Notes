import cv2
import numpy as np
import os.path

"""
NOTE
If no video loads after running code, you may need to copy your opencv_ffmpeg.dll
From: C:\opencv2413\opencv\sources\3rdparty\ffmpeg
To: Where your python is installed e.g. C:\Anaconda2\ \
Once it's copied you'll need to rename the file according to the version of OpenCV you're using.
e.g. if you're using OpenCV 2.4.13 then rename the file as:
opencv_ffmpeg2413_64.dll or opencv_ffmpeg2413.dll (if you're using an X86 machine)
opencv_ffmpeg310_64.dll or opencv_ffmpeg310.dll (if you're using an X86 machine)
"""

# Create our body classifier
body_classifier = cv2.CascadeClassifier(os.path.dirname(__file__) + '/../Haarcascades/haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture(os.path.dirname(__file__) + '/../images/walking.avi')

# Loop once video is successfully loaded
while cap.isOpened():
    
    # Read first frame
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Pass frame to our body classifier
        bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
        
        # Extract bounding boxes for any bodies identified
        for (x,y,w,h) in bodies:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.imshow('Pedestrians', frame)

        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break
    else:
        break

print 'End of video'
cv2.waitKey()
cap.release()
cv2.destroyAllWindows()