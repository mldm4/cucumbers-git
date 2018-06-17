import cv2
import numpy as np

def skeletonize(img):

    # size = np.size(img)
    # skel = np.zeros(img.shape, np.uint8)
    #
    # ret, img = cv2.threshold(img, 127, 255, 0)
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # done = False
    #
    # while (not done):
    #     eroded = cv2.erode(img, element)
    #     temp = cv2.dilate(eroded, element)
    #     temp = cv2.subtract(img, temp)
    #     skel = cv2.bitwise_or(skel, temp)
    #     img = eroded.copy()
    #
    #     zeros = size - cv2.countNonZero(img)
    #     if zeros == size:
    #         done = True

    img = img.copy()  # don't clobber original
    skel = img.copy()

    skel[:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:, :] = eroded[:, :]
        if cv2.countNonZero(img) == 0:
            break

    cv2.imshow("skel", skel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imBGR = cv2.imread('bbox/cucumber_belt_537_bbox.jpg') #525, 518, 537
'''GESTIONAR COLOR DE IMAGEN'''
# imB, imG, imR = imBGR[:,:,0], imBGR[:,:,1], imBGR[:,:,2]
im = cv2.cvtColor(imBGR, cv2.COLOR_BGR2GRAY)
# cv2.imshow('B', imB)
# cv2.waitKey()
# cv2.imshow('G', imG)
# cv2.waitKey()
# cv2.imshow('R', imR)
# cv2.waitKey()
# im = imG

# im_canny = cv2.Canny(im, 200, 255)
# im_sobelx = cv2.Sobel(im, cv2.CV_16S, 1, 0)
# im_sobely = cv2.Sobel(im, cv2.CV_16S, 0, 1)
# im_sobel = cv2.addWeighted(im_sobelx, 0.5, im_sobely, 0.5, 0)
'''BINARIZAR IMAGEN'''
thr, im_thr = cv2.threshold(im, 230, 255, cv2.THRESH_BINARY)
# thr_otsu, im_otsu = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('canny', im_canny)
# cv2.waitKey()
# cv2.imshow('sobel', im_thr)
# cv2.waitKey()
print(thr)
im_thr[np.where(im_thr == 0)] = 1
im_thr[np.where(im_thr == 255)] = 0
im_thr[np.where(im_thr == 1)] = 255
cv2.imshow('binary', im_thr)
cv2.waitKey()

'''CERRAR IMAGEN BINARIA'''
e_element = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
# im_eroded = cv2.erode(im_canny, e_element)
im_opened = cv2.morphologyEx(im_thr, cv2.MORPH_CLOSE, e_element)
# cv2.imshow('eroded', im_eroded)
# cv2.waitKey()
cv2.imshow('opened', im_opened)
cv2.waitKey()

'''ENCONTRAR CONTORNOS'''
im_cont, contours, hierarchy = cv2.findContours(im_thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (0,255,0), 3)
cv2.imshow('cont', im)
cv2.waitKey()

cv2.destroyAllWindows()

im_patBGR = cv2.imread('bbox/cucumber_belt_509_bbox.jpg')
im_pat = cv2.cvtColor(im_patBGR, cv2.COLOR_BGR2GRAY)
thr, im_pattern = cv2.threshold(im_pat, 230, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('binary', im_pattern)
cv2.waitKey()
im_pattern, pattern_contours, hierarchy = cv2.findContours(im_pattern,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im_pat, pattern_contours, -1, (0,255,0), 3)
cv2.imshow('cont', im_pat)
cv2.waitKey()

#contour attribs
ellipse = cv2.fitEllipse(contours[0])
cv2.ellipse(imBGR,ellipse,(0,255,0),2)
cv2.imshow('cont', imBGR)
cv2.waitKey()

#sketeize, quitar ramas y luego dilate a tope
skeletonize(im_thr)

ret = cv2.matchShapes(contours[0],pattern_contours[0],1,0.0)
print(ret)

cv2.destroyAllWindows()

