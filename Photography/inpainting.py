"""
What is Computational Photography?

- Digital image processing techniques used on images produced by cameras
- Seeks to enhance images via processing rather than use expensive optical processes
- Used significantly in all cameras, especially smartphones:
    - Noise Reduction
    - High Dynamic Range
    - Image stabilization
    - Panoramas
    - Inpainting (removal of small noises, strokes etc.)

Inpainting is the process of reconstructing lost or deteriorated parts of images and videos. 
It is an advanced form of interpolation that can be used to replace lost or corrupted parts 
of the image data.

cv2.inpaint(input image, mask, inpaintRadius, Inpaint Method)

inpaintRadius – Radius of a circular neighborhood of each point inpainted that is considered 
by the algorithm. Smaller values look less blurred, while larger values look more pixelated or 
blurred.

Inpaint Methods:
INPAINT_NS - Navier-Stokes based method [Navier01]
INPAINT_TELEA - Method by Alexandru Telea [Telea04] - Better as it integrates more seamlessly 
into the image
"""
import cv2
import numpy as np
import os.path

# Load our damaged photo
dirname = os.path.dirname(__file__) + '/../'
image = cv2.imread(dirname + 'images/abraham.jpg')
cv2.imshow('Original Damaged Photo', image)
cv2.waitKey(0)

# Load the photo where we've marked the damaged areas
marked_damages = cv2.imread(dirname + 'images/mask.jpg', 0)
cv2.imshow('Marked Damages', marked_damages)
cv2.waitKey(0)

# Let's make a mask out of our marked image be changing all colors 
# that are not white, to black
ret, thresh1 = cv2.threshold(marked_damages, 254, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', thresh1)
cv2.waitKey(0)

# Let's dilate (make thicker) our the marks w made
# since thresholding has narrowed it slightly
kernel = np.ones((7,7), np.uint8)
mask = cv2.dilate(thresh1, kernel, iterations = 1)
cv2.imshow('Dilated Mask', mask)
cv2.imwrite(dirname + 'images/abraham_mask.png', mask)

cv2.waitKey(0)
restored = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

cv2.imshow('Restored', restored)
cv2.waitKey(0)
cv2.destroyAllWindows()
