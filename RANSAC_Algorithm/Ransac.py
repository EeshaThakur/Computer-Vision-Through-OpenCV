import numpy as np
from Affine import estimate_affine

K = 3
THRESHOLD = 5
ITER_NUM = 2000


def residual_lengths(A, B, s, t):
    """
    Compute residual error.
    """

    estimated = np.dot(A, s) + B

    diff = estimated - t

    residual = np.sqrt(np.sum(diff ** 2, axis=0))

    return residual


def ransac_fit(pts_s, pts_t):

    best_inliers = None
    best_A = None
    best_B = None
    best_count = 0

    n = pts_s.shape[1]

    if n < K:
        return None, None, None

    for _ in range(ITER_NUM):

        idx = np.random.choice(n, K, replace=False)

        try:
            A_tmp, B_tmp = estimate_affine(
                pts_s[:, idx],
                pts_t[:, idx]
            )
        except:
            continue

        residual = residual_lengths(
            A_tmp,
            B_tmp,
            pts_s,
            pts_t
        )

        inliers = np.where(residual < THRESHOLD)[0]

        if len(inliers) > best_count:
            best_count = len(inliers)
            best_inliers = inliers
            best_A = A_tmp
            best_B = B_tmp

    return best_A, best_B, best_inliers