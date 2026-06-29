import numpy as np

def estimate_affine(s, t):
    """
    Estimate affine transformation:
    t = A*s + b

    s : 2 x N source points
    t : 2 x N target points
    """

    num = s.shape[1]

    if num < 3:
        raise ValueError("At least 3 points are required.")

    M = np.zeros((2 * num, 6))

    for i in range(num):
        x = s[0, i]
        y = s[1, i]

        M[2 * i] = [x, y, 0, 0, 1, 0]
        M[2 * i + 1] = [0, 0, x, y, 0, 1]

    b = t.T.reshape((2 * num, 1))

    theta, _, _, _ = np.linalg.lstsq(M, b, rcond=None)

    A = theta[:4].reshape((2, 2))
    B = theta[4:].reshape((2, 1))

    return A, B