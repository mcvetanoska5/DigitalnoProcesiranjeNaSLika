import cv2

image = cv2.imread('database/13567.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshold_value = 150
ret, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', binary_image)
cv2.imshow('Processed Image', morph_image)
cv2.imshow('Contour Image', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()