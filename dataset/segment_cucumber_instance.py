import cv2

imBGR = cv2.imread('bbox/cucumber_2_bbox.jpg')
imB, imG, imR = imBGR[:,:,0], imBGR[:,:,1], imBGR[:,:,2]
im = cv2.cvtColor(imBGR, cv2.COLOR_BGR2GRAY)
# cv2.imshow('B', imB)
# cv2.waitKey()
# cv2.imshow('G', imG)
# cv2.waitKey()
# cv2.imshow('R', imR)
# cv2.waitKey()
im = imG

im_canny = cv2.Canny(im, 200, 255)
im_sobelx = cv2.Sobel(im, cv2.CV_8U, 1, 0)
im_sobely = cv2.Sobel(im, cv2.CV_8U, 0, 1)
im_sobel = cv2.addWeighted(im_sobelx, 0.5, im_sobely, 0.5, 0)
thr, im_thr = cv2.threshold(im_sobel, 100, 255, cv2.THRESH_BINARY)
thr_otsu, im_otsu = cv2.threshold(im_sobel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('canny', im_canny)
cv2.waitKey()
cv2.imshow('sobel', im_thr)
cv2.waitKey()
cv2.imshow('otsu', im_otsu)
cv2.waitKey()

e_element = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
im_eroded = cv2.erode(im_canny, e_element)
im_opened = cv2.morphologyEx(im_canny, cv2.MORPH_OPEN, e_element)
# cv2.imshow('eroded', im_eroded)
# cv2.waitKey()
# cv2.imshow('opened', im_opened)
# cv2.waitKey()

cv2.destroyAllWindows()
