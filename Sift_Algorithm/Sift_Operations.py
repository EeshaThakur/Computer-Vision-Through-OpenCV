import cv2
import matplotlib.pyplot as plt

def extract_sift_features(img):
    """
    Detect SIFT keypoints and descriptors
    """

    # Create SIFT object
    sift = cv2.SIFT_create()

    key_points, descriptors = sift.detectAndCompute(img, None)

    return key_points, descriptors


def showing_sift_features(gray_img, original_img, key_points):
    """
    Display SIFT keypoints
    """

    image_with_keypoints = cv2.drawKeypoints(
        gray_img,
        key_points,
        original_img.copy(),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title("SIFT Keypoints")
    plt.axis("off")
    plt.show()
    