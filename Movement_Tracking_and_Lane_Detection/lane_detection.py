import cv2
import numpy as np

# ---------------------------------------------------
# REGION OF INTEREST
# ---------------------------------------------------
def region_selection(image):

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    rows, cols = image.shape[:2]

    bottom_left = [int(cols * 0.1), int(rows * 0.95)]
    top_left = [int(cols * 0.4), int(rows * 0.6)]
    top_right = [int(cols * 0.6), int(rows * 0.6)]
    bottom_right = [int(cols * 0.9), int(rows * 0.95)]

    vertices = np.array(
        [[bottom_left, top_left, top_right, bottom_right]],
        dtype=np.int32
    )

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


# ---------------------------------------------------
# HOUGH TRANSFORM
# ---------------------------------------------------
def hough_transform(image):

    rho = 1
    theta = np.pi / 180
    threshold = 20
    min_line_length = 20
    max_line_gap = 300

    lines = cv2.HoughLinesP(
        image,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    return lines


# ---------------------------------------------------
# AVERAGE SLOPE INTERCEPT
# ---------------------------------------------------
def average_slope_intercept(lines):

    left_lines = []
    left_weights = []

    right_lines = []
    right_weights = []

    if lines is None:
        return None, None

    for line in lines:

        x1, y1, x2, y2 = line.reshape(4)

        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        length = np.sqrt(
            (y2 - y1) ** 2 +
            (x2 - x1) ** 2
        )

        if slope < 0:
            left_lines.append((slope, intercept))
            left_weights.append(length)
        else:
            right_lines.append((slope, intercept))
            right_weights.append(length)

    left_lane = None
    right_lane = None

    if len(left_lines) > 0:
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights)

    if len(right_lines) > 0:
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights)

    return left_lane, right_lane


# ---------------------------------------------------
# PIXEL POINTS
# ---------------------------------------------------
def pixel_points(y1, y2, line):

    if line is None:
        return None

    slope, intercept = line

    if abs(slope) < 0.001:
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return ((x1, int(y1)), (x2, int(y2)))


# ---------------------------------------------------
# LANE LINES
# ---------------------------------------------------
def lane_lines(image, lines):

    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0]
    y2 = int(y1 * 0.6)

    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)

    return left_line, right_line


# ---------------------------------------------------
# DRAW LANE LINES
# ---------------------------------------------------
def draw_lane_lines(image, lines,
                    color=(0, 255, 0),
                    thickness=10):

    line_image = np.zeros_like(image)

    for line in lines:

        if line is not None:

            cv2.line(
                line_image,
                line[0],
                line[1],
                color,
                thickness
            )

    return cv2.addWeighted(
        image,
        0.8,
        line_image,
        1.0,
        0
    )


# ---------------------------------------------------
# FRAME PROCESSOR
# ---------------------------------------------------
def frame_processor(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    roi = region_selection(edges)

    lines = hough_transform(roi)

    if lines is None:
        return image

    lanes = lane_lines(image, lines)

    result = draw_lane_lines(
        image,
        lanes
    )

    return result


# ---------------------------------------------------
# VIDEO PROCESSING
# ---------------------------------------------------
def process_video(input_video, output_video):

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("Cannot open video file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = frame_processor(frame)

        out.write(result)

        cv2.imshow("Lane Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":

    input_video = "input.mp4"
    output_video = "output.mp4"

    process_video(input_video, output_video)

    print("Lane Detection Completed")
    print("Output saved as output.mp4")