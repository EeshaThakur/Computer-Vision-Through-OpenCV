import cv2
import numpy as np
from collections import deque

# Store previous points
pts = deque(maxlen=128)

# Green color range in HSV
Lower_green = np.array([40, 50, 50])
Upper_green = np.array([90, 255, 255])

# Start webcam
cap = cv2.VideoCapture(0)

# Check webcam
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Set webcam resolution
cap.set(3, 640)
cap.set(4, 480)

# Kernel for noise removal
kernel = np.ones((5, 5), np.uint8)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for green color
    mask = cv2.inRange(hsv, Lower_green, Upper_green)

    # Remove noise
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    center = None

    if len(contours) > 0:

        # Largest contour
        c = max(contours, key=cv2.contourArea)

        ((x, y), radius) = cv2.minEnclosingCircle(c)

        M = cv2.moments(c)

        if M["m00"] != 0:

            center = (
                int(M["m10"] / M["m00"]),
                int(M["m01"] / M["m00"])
            )

            if radius > 10:

                # Draw tracking circle
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    int(radius),
                    (0, 255, 255),
                    2
                )

                # Draw center point
                cv2.circle(
                    frame,
                    center,
                    5,
                    (0, 0, 255),
                    -1
                )

                pts.appendleft(center)

    # Draw movement trail
    for i in range(1, len(pts)):

        if pts[i - 1] is None or pts[i] is None:
            continue

        cv2.line(
            frame,
            pts[i - 1],
            pts[i],
            (0, 0, 255),
            3
        )

    # Display final output only
    cv2.imshow("Movement Tracking", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press ESC to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()