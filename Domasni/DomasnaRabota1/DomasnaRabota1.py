import cv2
import numpy as np

# Определување на функцијата за contrast stretching
def contrast_stretching(image, points):
    rows, cols, channels = image.shape
    stretched_img = np.zeros_like(image)
    for channel in range(channels):
        channel_points = points[channel]
        min_val = min(channel_points)
        max_val = max(channel_points)
        diff = max_val - min_val
        if diff == 0:
            diff = 1
        multiplier = 255.0 / diff
        stretched_channel = (image[:, :, channel] - min_val) * multiplier
        stretched_channel = np.clip(stretched_channel, 0, 255).astype(np.uint8)
        stretched_img[:, :, channel] = stretched_channel
    return stretched_img


# Вчитување на сликата
image = cv2.imread(r'C:\Users\user\Desktop\func_transform.png')
points = [[50, 200], [20, 230], [10, 210]]

# Примена на contrast stretching
stretched_img = contrast_stretching(image, points)

# Прикажување на оригиналната и трансформираната слика
cv2.imshow('Original image', image)
cv2.imshow('Stretched image', stretched_img)

# Чекање за притискање на копче за затворање на прозорецот
cv2.waitKey()
