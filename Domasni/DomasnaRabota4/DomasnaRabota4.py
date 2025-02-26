import cv2
import os
def similarity(image1, image2):
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    _, thresh_image1 = cv2.threshold(gray_image1, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, thresh_image2 = cv2.threshold(gray_image2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours_image1, _ = cv2.findContours(thresh_image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_image2, _ = cv2.findContours(thresh_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    similarity = cv2.matchShapes(contours_image1[0], contours_image2[0], cv2.CONTOURS_MATCH_I1, 0)
    return similarity
def similar_images(query_dir, database_dir):
    query = []
    database = []
    for filename in os.listdir(query_dir):
        image_path = os.path.join(query_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            query.append(image)
    i = 1
    for query_image in query:
        similarities = []
        for filename in os.listdir(database_dir):
            image_path = os.path.join(database_dir, filename)
            database_image = cv2.imread(image_path)
            if database_image is not None:
                similar = similarity(query_image, database_image)
                similarities.append((filename, similar))
        sorted_images = sorted(similarities, key=lambda x: x[1])
        print("Slika {}".format(i))
        i += 1
        for image_name, similar in sorted_images:
            print(image_name, ":", similar)
        print()
query_dir = "C:/Users/user/Downloads/query_images"
database_dir = "C:/Users/user/Downloads/database"
similar_images(query_dir, database_dir)