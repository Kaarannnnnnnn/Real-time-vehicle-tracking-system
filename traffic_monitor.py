import cv2
import numpy as np
from tracker import *

#mask creation:
def create_mask_zones(height, width):
    mask = np.zeros((height, width), dtype=np.uint8)

    road_start_y = int(height * 0.15)
    mask[road_start_y:height, :] = 255

    # road perspective mask
    road_polygon = np.array([
        [int(width * 0.1), road_start_y],  # Top left of road
        [int(width * 0.9), road_start_y],  # Top right of road
        [width, height],  # Bottom right
        [0, height]  # Bottom left
    ], np.int32)

    cv2.fillPoly(mask, [road_polygon], 255)
    return mask


def create_dynamic_mask(height, width):
    #mask for road
    mask = np.zeros((height, width), dtype=np.uint8)

    #road coverage
    top_y = int(height * 0.12)

    points = np.array([
        [int(width * 0.25), top_y],  # Top left of road
        [int(width * 0.75), top_y],  # Top right of road
        [width, height],  # Bottom right
        [0, height]  # Bottom left
    ], np.int32)

    cv2.fillPoly(mask, [points], 255)
    return mask


def clean_mask(mask):
   #noise reduction from mask
    # Remove small noise blobs
    contours_temp, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_clean = np.zeros_like(mask)

    for cnt in contours_temp:
        area = cv2.contourArea(cnt)
        if area > 150:  # Remove very small noise
            cv2.fillPoly(mask_clean, [cnt], 255)

    return mask_clean


def filter_detections_by_position(detections, height, width):
    #more noise reduction by filtering
    filtered = []


    min_y = int(height * 0.12)  # Reduced from 0.25
    max_y = int(height * 0.98)  # Increased from 0.95
    min_x = int(width * 0.02)  # Reduced from 0.05
    max_x = int(width * 0.98)  # Increased from 0.95

    for detection in detections:
        x, y, w, h = detection
        center_x = x + w // 2
        center_y = y + h // 2


        if (min_x < center_x < max_x and
                min_y < center_y < max_y):
            filtered.append(detection)

    return filtered


def main():
    tracker = EuclideanDistTracker()
    cap = cv2.VideoCapture("highway.mp4.mp4")


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video dimensions: {width}x{height}")


    road_mask = create_dynamic_mask(height, width)
    # road_mask = create_mask_zones(height, width)

    # Optimized background subtractor
    object_detector = cv2.createBackgroundSubtractorMOG2(
        history=500,  # Increased for more stable background
        varThreshold=25,  # Reduced for higher sensitivity
        detectShadows=True
    )

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or failed to read frame")
            break

        frame_count += 1
        original_frame = frame.copy()


        frame_masked = cv2.bitwise_and(frame, frame, mask=road_mask)

        # Object Detection
        mask = object_detector.apply(frame_masked)


        _, mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)

        # Adaptive morphological operations
        kernel_size = max(5, min(11, width // 150))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


        mask = cv2.medianBlur(mask, 7)


        mask = clean_mask(mask)


        mask = cv2.bitwise_and(mask, road_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        #  adaptive thresholds
        min_area = max(400, (width * height) // 1500)  # Reduced minimum area
        min_width = max(30, width // 35)  # Reduced minimum width
        min_height = max(25, height // 30)  # Reduced minimum height
        max_area = (width * height) // 15  # Increased maximum area

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)


                if w > min_width and h > min_height:
                    aspect_ratio = w / h
                    extent = area / (w * h)


                    if (0.2 < aspect_ratio < 6.0 and  # Wider aspect ratio range
                            extent > 0.15 and  # Lower fill ratio requirement
                            w < width // 2 and  # More lenient width limit
                            h < height // 2):  # More lenient height limit
                        detections.append([x, y, w, h])

        # Filter detections by position
        detections = filter_detections_by_position(detections, height, width)

        # Object Tracking
        boxes_id = tracker.update(detections)

        for box_id in boxes_id:
            x, y, w, h, vehicle_id = box_id

            #  bounding box with thicker lines
            cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # vehicle ID with background
            label = f"ID: {vehicle_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(original_frame, (x, y - label_size[1] - 10),
                          (x + label_size[0], y), (0, 255, 0), -1)
            cv2.putText(original_frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            center = (x + w // 2, y + h // 2)
            cv2.circle(original_frame, center, 5, (0, 0, 255), -1)

        # Draw statistics with background
        stats_text = f"Vehicles: {len(boxes_id)} | Frame: {frame_count}"
        text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(original_frame, (5, 5), (text_size[0] + 15, text_size[1] + 15), (0, 0, 0), -1)
        cv2.putText(original_frame, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Create side-by-side comparison for better visualization
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        road_mask_colored = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)

        # Resize for display if frame is too large
        display_height = 400
        if height > display_height:
            scale = display_height / height
            new_width = int(width * scale)
            original_frame_resized = cv2.resize(original_frame, (new_width, display_height))
            mask_colored_resized = cv2.resize(mask_colored, (new_width, display_height))
            road_mask_colored_resized = cv2.resize(road_mask_colored, (new_width, display_height))
        else:
            original_frame_resized = original_frame
            mask_colored_resized = mask_colored
            road_mask_colored_resized = road_mask_colored

        # Display windows
        cv2.imshow("Vehicle Tracking", original_frame_resized)
        cv2.imshow("Detection Mask", mask_colored_resized)

        # Show road mask for first 20 frames to verify it's working
        if frame_count <= 20:
            cv2.imshow("Road Mask", road_mask_colored_resized)

        # Controls
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            object_detector = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=25, detectShadows=True)
            print("Background model reset")
        elif key == ord('p'):
            cv2.waitKey(0)
        elif key == ord('s'):
            cv2.imwrite(f'vehicle_detection_frame_{frame_count}.jpg', original_frame)
            print(f"Screenshot saved: vehicle_detection_frame_{frame_count}.jpg")

    print(f"\nProcessing completed. Total frames: {frame_count}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()