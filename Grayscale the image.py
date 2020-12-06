# Grayscale the image
gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray_img',gray_img)
# Save image
cv.imwrite('gray_lena.jpg',gray_img)
cv.waitKey(0)
cv.destroyAllWindows()
