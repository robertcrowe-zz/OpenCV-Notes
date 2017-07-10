from __future__ import print_function
import cv2
import numpy as np
import os.path

# Grayscale and Canny Edges extracted
image = cv2.imread(os.path.dirname(__file__) + '/../images/soduku.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 170, apertureSize = 3)

# Again we use the same rho and theta accuracies
# However, we specify a minimum vote (pts along line) of 100
# and min line length of 5 pixels and max gap between lines of 10 pixels
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, 5, 10)
print(lines.shape)

for x1, y1, x2, y2 in lines[0]:
    cv2.line(image, (x1, y1), (x2, y2),(0, 255, 0), 3)

cv2.imshow('Probabilistic Hough Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()