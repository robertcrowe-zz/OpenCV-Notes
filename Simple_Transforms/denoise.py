import numpy as np
import cv2
import os.path

"""
There are 4 variations of Non-Local Means Denoising:
cv2.fastNlMeansDenoising() - works with a single grayscale images
cv2.fastNlMeansDenoisingColored() - works with a color image.
cv2.fastNlMeansDenoisingMulti() - works with image sequence captured in short period of time (grayscale images)
cv2.fastNlMeansDenoisingColoredMulti() - same as above, but for color images.
"""
image = cv2.imread(os.path.dirname(__file__) + '/../images/elephant.jpg')
cv2.imshow('Original', image)
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Parameters, after None are - the filter strength 'h' (5-10 is a good range)
# Next is hForColorComponents, set as same value as h again
"""
Python: cv2.fastNlMeansDenoising(src[, dst[, h[, templateWindowSize[, searchWindowSize]]]]) -> dst
Parameters:	
src – Input 8-bit 1-channel, 2-channel or 3-channel image.
dst – Output image with the same size and type as src .
templateWindowSize – Size in pixels of the template patch that is used to compute weights. 
Should be odd. Recommended value 7 pixels
searchWindowSize – Size in pixels of the window that is used to compute weighted average for 
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater 
denoising time. Recommended value 21 pixels
h – Parameter regulating filter strength. Big h value perfectly removes noise but also removes 
image details, smaller h value preserves details but also preserves some noise
This function expected to be applied to grayscale images. For colored images look at 
fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of 
colored image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored 
by converting image to CIELAB colorspace and then separately denoise L and AB components with 
different h parameter.
"""
dst0 = cv2.fastNlMeansDenoising(gray, None, 6, 7, 21)
cv2.imshow('Fast Means Denoising', dst0)
cv2.waitKey(0)

"""
Python: cv2.fastNlMeansDenoisingColored(src[, dst[, h[, hColor[, templateWindowSize[, searchWindowSize]]]]]) -> dst
Parameters:	
src – Input 8-bit 3-channel image.
dst – Output image with the same size and type as src .
templateWindowSize – Size in pixels of the template patch that is used to compute weights. 
Should be odd. Recommended value 7 pixels
searchWindowSize – Size in pixels of the window that is used to compute weighted average for 
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater 
denoising time. Recommended value 21 pixels
h – Parameter regulating filter strength for luminance component. Bigger h value perfectly 
removes noise but also removes image details, smaller h value preserves details but also 
preserves some noise
hForColorComponents – The same as h but for color components. For most images value equals 
10 will be enought to remove colored noise and do not distort colors
The function converts image to CIELAB colorspace and then separately denoise L and AB 
components with given h parameters using fastNlMeansDenoising function.
"""
dst1 = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
cv2.imshow('Fast Means Denoising (color)', dst1)
cv2.waitKey(0)

"""
Python: cv2.fastNlMeansDenoisingMulti(srcImgs, imgToDenoiseIndex, temporalWindowSize[, dst[, h[, templateWindowSize[, searchWindowSize]]]]) -> dst
Parameters:	
srcImgs – Input 8-bit 1-channel, 2-channel or 3-channel images sequence. All images 
should have the same type and size.
imgToDenoiseIndex – Target image to denoise index in srcImgs sequence
temporalWindowSize – Number of surrounding images to use for target image denoising. 
Should be odd. 
Images from imgToDenoiseIndex - temporalWindowSize / 2 to imgToDenoiseIndex - temporalWindowSize / 2 
from srcImgs will be used to denoise srcImgs[imgToDenoiseIndex] image.
dst – Output image with the same size and type as srcImgs images.
templateWindowSize – Size in pixels of the template patch that is used to compute weights. 
Should be odd. Recommended value 7 pixels
searchWindowSize – Size in pixels of the window that is used to compute weighted average 
for given pixel. Should be odd. Affect performance linearly: 
    greater searchWindowsSize - greater denoising time. Recommended value 21 pixels
h – Parameter regulating filter strength for luminance component. Bigger h value perfectly 
removes noise but also removes image details, smaller h value preserves details but also 
preserves some noise
"""
dst2 = cv2.fastNlMeansDenoisingMulti([gray], 0, 1, None, 6, 7, 21)
cv2.imshow('Fast Means Denoising Multi', dst2)
cv2.waitKey(0)

dst3 = cv2.fastNlMeansDenoisingColoredMulti([image], 0, 1, None, 6, 6, 7, 21)
cv2.imshow('Fast Means Denoising Multi (color)', dst3)
cv2.waitKey(0)

cv2.destroyAllWindows()