import cv2
import numpy as np

def compass_edge_detector(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    compass_operators = [np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
                         np.array([[-1, -1,  2], [-1,  2, -1], [ 2, -1, -1]])]
    edge_responses = [cv2.filter2D(gradient_x, cv2.CV_64F, kernel) +
                      cv2.filter2D(gradient_y, cv2.CV_64F, kernel)
                      for kernel in compass_operators]
    return edge_responses

def combine_edges(edge_responses, threshold):
    # Non-maximum suppression
    non_max_suppressed = np.max(edge_responses, axis=0)
    edges = non_max_suppressed > threshold
    return edges
image_path = r'C:\Users\user\Desktop\slika.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
edges = compass_edge_detector(original_image)
threshold_values = [50,70,90]
combined_image = np.zeros_like(original_image)
for idx, threshold in enumerate(threshold_values, start=1):
    combined_edges = combine_edges(edges, threshold)
    combined_image[combined_edges] = 255 * (idx / len(threshold_values))
cv2.imshow('Combined Edges', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
