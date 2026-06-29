import cv2
import numpy as np

from Align import (
    extract_SIFT,
    match_SIFT,
    affine_matrix
)

img_source = cv2.imread("2.jpg")
img_target = cv2.imread("target.png")

if img_source is None:
    raise ValueError("Cannot read source.jpg")

if img_target is None:
    raise ValueError("Cannot read target.png")

kp_source, desc_source = extract_SIFT(img_source)

kp_target, desc_target = extract_SIFT(img_target)

pos = match_SIFT(
    desc_source,
    desc_target
)

print("Matches Found :", len(pos))

H = affine_matrix(
    kp_source,
    kp_target,
    pos
)

rows, cols = img_target.shape[:2]

warp = cv2.warpAffine(
    img_source,
    H,
    (cols, rows)
)

merge = cv2.addWeighted(
    img_target,
    0.5,
    warp,
    0.5,
    0
)
cv2.imwrite("registered_output.jpg", merge)
print("Image saved successfully!")
cv2.imshow("Registered Image", merge)

cv2.waitKey(0)
cv2.destroyAllWindows()