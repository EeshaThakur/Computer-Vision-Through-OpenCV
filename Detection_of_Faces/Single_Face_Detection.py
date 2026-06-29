import cv2

# Read image
img = cv2.imread("single_face.jpg")

# Check if image loaded successfully
if img is None:
    print("Error: Image not found!")
    exit()


# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)

# Detect faces
faces = face_cascade.detectMultiScale(
    img,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

# Show detected image
cv2.imshow("Final Detected Image", img)

# Save output image
cv2.imwrite("detected_faces.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
