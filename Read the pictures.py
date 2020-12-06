# Import module
import cv2 as cv
# Read image
img=cv.imread('lena.jpg')
# Display image
cv.imshow('read_img',img)
# Wait for keyboard input in milliseconds
cv.waitKey(3000)
# Free memory
cv.destroyAllWindows()


