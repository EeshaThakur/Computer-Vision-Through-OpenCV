import cv2
import numpy as np
import matplotlib.pyplot as plt
from Sift_Operations import *

print("Make sure both images are present in the same folder")

# Read first image
img1_name = input("Enter First Image Name: ")
Image1 = cv2.imread(img1_name)

# Read second image
img2_name = input("Enter Second Image Name: ")
Image2 = cv2.imread(img2_name)

# Check images loaded successfully
if Image1 is None:
    print("Error: First image not found!")
    exit()

if Image2 is None:
    print("Error: Second image not found!")
    exit()

# Convert images to grayscale
Image1_gray = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)
Image2_gray = cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY)

# Extract SIFT features
Image1_key_points, Image1_descriptors = extract_sift_features(Image1_gray)
Image2_key_points, Image2_descriptors = extract_sift_features(Image2_gray)

print("Number of Keypoints in Image 1 =", len(Image1_key_points))
print("Number of Keypoints in Image 2 =", len(Image2_key_points))

# Show SIFT keypoints
showing_sift_features(
    Image1_gray,
    Image1,
    Image1_key_points
)

showing_sift_features(
    Image2_gray,
    Image2,
    Image2_key_points
)

# Create BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(
    Image1_descriptors,
    Image2_descriptors
)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

print("Total Matches Found =", len(matches))

# Draw best 100 matches
matched_image = cv2.drawMatches(
    Image1,
    Image1_key_points,
    Image2,
    Image2_key_points,
    matches[:100],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Display matched image
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title("Top 100 SIFT Matches")
plt.axis("off")
plt.show()

# Save output
cv2.imwrite("SIFT_Matches.jpeg", matched_image)

print("Output saved as SIFT_Matches.jpeg")