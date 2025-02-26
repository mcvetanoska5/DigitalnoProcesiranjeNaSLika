import cv2
import numpy as np

img = cv2.imread('database/10005.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = np.zeros_like(img)
cv2.drawContours(contour_img, contours, -1, (0, 140, 0), 2)
cv2.imshow('Original Image', img)
cv2.imshow('Segmentation', thresh)
processed_img = cv2.medianBlur(thresh, 5)
cv2.imshow('Processed Image', processed_img)
cv2.imshow('Image with Contours', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()