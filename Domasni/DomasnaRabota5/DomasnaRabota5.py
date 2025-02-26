import glob
import cv2
import numpy as np
FLANN_INDEX_KDTREE = 0
def find_keypoints(img_src):
    grey = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.SIFT_create()
    kp, ds = descriptor.detectAndCompute(grey, None)
    return kp, ds
def find_matches(des1, des2):
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches
def open_imgs(dir_path):
    data_path = cv2.os.path.join(dir_path, '*.jpg')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        img = cv2.imread(f1)
        data.append(img)
    return data
def get_img_hash(img):
    return repr(hash(img.tobytes()))
def describe_posters(posters):
    kps_and_dsc = {}
    for poster in posters:
        kps_and_dsc[get_img_hash(poster)] = find_keypoints(poster)
    return kps_and_dsc
def index_posters(posters):
    p_index = {}
    for poster in posters:
        p_index[get_img_hash(poster)] = poster
    return p_index
def find_inliers(kp1, kp2, good_matches):
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        inliers = np.sum(matchesMask)
    else:
        inliers = 0
        matchesMask = None
    return inliers, matchesMask
def draw_matches(img1, kp1, img2, kp2, matches, matchesMask=None):
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    return result
if __name__ == '__main__':
    posters = open_imgs("Database")
    posters_index = index_posters(posters)
    poster_kps_and_dsc = describe_posters(posters)
    for i in range(1, 4):
        query_img = cv2.imread(f'C:/Users/user/PycharmProjects/DomasnaRabota5/hw7_poster_{i}.jpg')
        if query_img is None:
            print(f"Error: Unable to read image 'hw7_poster_{i}.jpg'. Check the file path.")
            continue
        query_img_resize = cv2.resize(query_img, (int(query_img.shape[1] * 0.2), int(query_img.shape[0] * 0.2)))
        kp_query, ds_query = find_keypoints(query_img_resize)
        all_matches = {}
        for poster_hash, poster_kps_dsc in poster_kps_and_dsc.items():
            good_matches = find_matches(poster_kps_dsc[1], ds_query)
            inliers, matchesMask = find_inliers(poster_kps_dsc[0], kp_query, good_matches)
            if inliers > 0:
                all_matches[poster_hash] = (good_matches, inliers, matchesMask)
        if not all_matches:
            print(f"No good matches found for query image hw7_poster_{i}.jpg")
            continue
        best_match_hash = sorted(all_matches.items(), key=lambda x: x[1][1], reverse=True)[0][0]
        best_poster = posters_index[best_match_hash]
        best_good_matches = all_matches[best_match_hash][0]
        best_matchesMask = all_matches[best_match_hash][2]
        kp_poster, _ = poster_kps_and_dsc[best_match_hash]
        result1 = draw_matches(query_img_resize, kp_query, best_poster, kp_poster, best_good_matches)
        result2 = draw_matches(query_img_resize, kp_query, best_poster, kp_poster, best_good_matches, best_matchesMask)
        cv2.imshow(f"Query {i} and Best Match", np.hstack((query_img_resize, best_poster)))
        cv2.imshow(f"Query {i} with SIFT Keypoints", result1)
        cv2.imshow(f"Query {i} with Inliers", result2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
