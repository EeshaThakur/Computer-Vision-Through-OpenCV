import cv2
import numpy as np

from Ransac import ransac_fit
from Affine import estimate_affine


def extract_SIFT(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        sift = cv2.SIFT_create()
    except:
        sift = cv2.xfeatures2d.SIFT_create()

    kp, desc = sift.detectAndCompute(gray, None)

    kp = np.array([p.pt for p in kp]).T

    return kp, desc


def match_SIFT(desc_source, desc_target):

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(
        desc_source,
        desc_target,
        k=2
    )

    pos = []

    for m, n in matches:

        if m.distance < 0.75 * n.distance:
            pos.append([m.queryIdx, m.trainIdx])

    return np.array(pos, dtype=np.int32)


def affine_matrix(keypoint_source,
                  keypoint_target,
                  pos):

    if len(pos) < 3:
        raise ValueError(
            "Not enough matches found."
        )

    s = keypoint_source[:, pos[:, 0]]
    t = keypoint_target[:, pos[:, 1]]

    A, B, inliers = ransac_fit(s, t)

    if inliers is None or len(inliers) < 3:
        raise ValueError(
            "RANSAC failed to find enough inliers."
        )

    s_in = s[:, inliers]
    t_in = t[:, inliers]

    A, B = estimate_affine(
        s_in,
        t_in
    )

    H = np.hstack((A, B))

    return H