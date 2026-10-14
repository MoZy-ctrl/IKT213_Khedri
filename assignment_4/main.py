import cv2
import numpy as np

from PIL import Image


def harris_corner_detection(reference_image):
    img = cv2.imread(reference_image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    img[dst>0.01*dst.max()] = [0, 0, 255]

    cv2.imwrite('harris.png', img)
    return img

#Using SIFT

def align_images(image_to_align, reference_image, max_features, good_match_precent):
    img_align = cv2.imread(image_to_align)
    img_ref = cv2.imread(reference_image)

    #Scaling down the images to same height
    scale = img_ref.shape[0] / img_align.shape[0]
    img_align = cv2.resize(img_align, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    gray_align = cv2.cvtColor(img_align, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    #Keypoint
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_align, None)
    kp2, des2 = sift.detectAndCompute(gray_ref, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < good_match_precent * n.distance:
                good.append(m)

    if len(good) < max_features:
        print(f"not enough good matches for {len(good)}/{max_features}")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = img_ref.shape[:2]
    aligned = cv2.warpPerspective(img_align, H, (w, h))

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask.ravel().tolist(), flags=2)
    img_matches = cv2.drawMatches(img_align, kp1, img_ref, kp2, good, None, **draw_params)

    cv2.imwrite('aligned.png', aligned)

    scale = 0.4
    matches = cv2.resize(img_matches, (0, 0), fx=scale, fy=scale)
    cv2.imwrite('matches.png', matches)

    return aligned, img_matches

if __name__ == '__main__':
    harris_corner_detection('reference_img.png')
    align_images('align_this.jpg', 'reference_img.png',max_features=10, good_match_precent=0.7)

    pages = [Image.open(p).convert('RGB') for p in ['harris.png', 'aligned.png', 'matches.png']]
    pages[0].save('assignment4.pdf', save_all=True, append_images=pages[1:])