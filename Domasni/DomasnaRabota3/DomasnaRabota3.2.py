import cv2

image = cv2.imread('database/10001.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, segmented_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(segmented_image, 30, 100)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (255), 2)
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()