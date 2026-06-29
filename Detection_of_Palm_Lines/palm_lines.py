import cv2

# Read image
image = cv2.imread("palm.jpg")

# Check if image is loaded
if image is None:
    print("Error: palm.jpg not found!")
    exit()

# Show and save original image
cv2.imshow("Palm", image)
cv2.imwrite("original_palm.jpg", image)
cv2.waitKey(0)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grayscale", gray)
cv2.imwrite("grayscale_palm.jpg", gray)
cv2.waitKey(0)

# Detect palm lines using Canny Edge Detector
edges = cv2.Canny(gray, 40, 55, apertureSize=3)

cv2.imshow("Edges in Palm", edges)
cv2.imwrite("edges_palm.jpg", edges)
cv2.waitKey(0)

# Invert colors so lines become black
edges = cv2.bitwise_not(edges)

cv2.imshow("Inverted Palm Lines", edges)
cv2.imwrite("palmlines.jpg", edges)
cv2.waitKey(0)

# Convert to BGR before blending
palmlines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Blend with original image
img = cv2.addWeighted(palmlines, 0.3, image, 0.7, 0)

# Display and save final result
cv2.imshow("Final Palm Line Detection", img)
cv2.imwrite("final_palm_output.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Images saved successfully!")