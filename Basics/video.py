import cv2
import os.path

cap = cv2.VideoCapture(os.path.dirname(__file__) + '/../images/cars.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# cap = cv2.VideoCapture(os.path.dirname(__file__) + '/../images/cars.avi')
# print cap.isOpened()   # True = read video successfully. False - fail to read video.

# # fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fourcc = cv2.cv.CV_FOURCC(*'XVID')
# out = cv2.VideoWriter("cars_out.avi", fourcc, 20.0, (640, 360))
# print out.isOpened()  # True = write out video successfully. False - fail to write out video.

# cap.release()
# out.release()


# cv2.VideoCapture(os.path.dirname(__file__) + '/../images/cars.avi')
# print cv2.isOpened()   # True = read video successfully. False - fail to read video.

# fourcc = cv2.cv.CV_FOURCC(*'XVID')
# out = cv2.VideoWriter('cars_out.avi',fourcc, 20.0, (640,360))
# print out.isOpened()  # True = write out video successfully. False - fail to write out video.

# cap.release()
# out.release()