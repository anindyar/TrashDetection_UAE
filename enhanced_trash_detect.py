import cv2
import json
import argparse
import easyocr
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import math

# -------------------------------
# Helper function to mock GPS data
# -------------------------------
def mock_gps(lat, lon, step, frame_num):
    """Generate mock GPS coordinates per frame."""
    return lat + step * frame_num, lon + step * frame_num

# -------------------------------
# Helper function to calculate distance in pixels
# -------------------------------
def calculate_pixel_distance(box1_center, box2_center):
    """Calculate Euclidean distance between two box centers in pixels."""
    x1, y1 = box1_center
    x2, y2 = box2_center
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# -------------------------------
# Helper function to convert pixels to feet
# -------------------------------
def pixels_to_feet(pixel_distance, drone_altitude_ft=100, frame_width=3840):
    """
    Convert pixel distance to real-world feet.

    Assumptions for DJI drone:
    - Default altitude: 100 feet (~30 meters)
    - Horizontal FOV: ~84 degrees (DJI standard)
    - Frame width: 3840 pixels (4K)

    Formula: real_distance = (pixel_distance / frame_width) * ground_coverage
    ground_coverage = 2 * altitude * tan(FOV/2)
    """
    fov_horizontal_deg = 84.0  # DJI standard FOV
    fov_horizontal_rad = math.radians(fov_horizontal_deg)

    # Calculate ground coverage width at given altitude
    ground_coverage_ft = 2 * drone_altitude_ft * math.tan(fov_horizontal_rad / 2)

    # Meters per pixel
    ft_per_pixel = ground_coverage_ft / frame_width

    # Convert pixel distance to feet
    real_distance_ft = pixel_distance * ft_per_pixel
    return real_distance_ft

# -------------------------------
# Helper function to get bounding box center
# -------------------------------
def get_box_center(bbox):
    """Get center point of bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# -------------------------------
# Helper function to check IoU (Intersection over Union)
# -------------------------------
def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

# -------------------------------
# Helper function to extract and recognize license plate
# -------------------------------
def extract_plate_number(frame, car_bbox, ocr_reader):
    """
    Extract license plate from car region using OCR.
    Returns: (plate_number, confidence, plate_detected)
    """
    try:
        x1, y1, x2, y2 = map(int, car_bbox)

        # Crop car region
        car_region = frame[y1:y2, x1:x2]

        if car_region.size == 0:
            return None, 0.0, False

        # Run OCR on car region
        results = ocr_reader.readtext(car_region, detail=1)

        if not results:
            return None, 0.0, False

        # Filter for plate-like text (alphanumeric, reasonable length)
        plate_candidates = []
        for (bbox, text, conf) in results:
            # Clean text: remove spaces and special chars
            clean_text = ''.join(c for c in text if c.isalnum() or c in ['-', ' '])

            # UAE plates typically: 3-6 characters/numbers
            if 3 <= len(clean_text.replace(' ', '')) <= 10 and conf > 0.3:
                plate_candidates.append((clean_text, conf))

        if plate_candidates:
            # Return the highest confidence plate
            best_plate = max(plate_candidates, key=lambda x: x[1])
            return best_plate[0].strip(), best_plate[1], True

        return None, 0.0, False

    except Exception as e:
        print(f"[WARNING] Plate extraction error: {e}")
        return None, 0.0, False

# -------------------------------
# Main Function
# -------------------------------
def main(video_path, base_lat, base_lon, step, output_video, output_json,
         conf_thresh=0.6, drone_altitude_ft=100, proximity_radius_ft=15):

    # Load YOLO models
    print("[INFO] Loading YOLOv8 models...")
    print(f"[INFO] Confidence threshold: {conf_thresh}")
    print(f"[INFO] Drone altitude: {drone_altitude_ft} feet")
    print(f"[INFO] Proximity radius: {proximity_radius_ft} feet")

    # Base model - detects everything (person, car, traffic light, etc.)
    print("[INFO] Loading base YOLOv8 model (COCO dataset - 80 classes)...")
    base_model = YOLO("weights/yolov8n.pt")

    # Waste detection model - detects waste items
    print("[INFO] Loading custom waste detection model (4 waste classes)...")
    waste_model = YOLO("runs/detect/waste_detect/weights/best.pt")

    # Initialize OCR reader
    print("[INFO] Initializing EasyOCR for license plate recognition...")
    print("[INFO] This may take a moment to download language models...")
    ocr_reader = easyocr.Reader(['en'], gpu=False)  # English only, can add 'ar' for Arabic

    # Define vehicle classes that might be near trash
    vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}

    # Define non-waste classes from COCO
    non_waste_classes = {
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    }

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

    # Prepare output video writer
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    detections_list = []

    # Checkpoint interval
    checkpoint_interval = 500
    checkpoint_file = output_json + ".checkpoint"

    print("[INFO] Processing video...")
    frame_num = 0

    try:
        for frame_num in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] Could not read frame {frame_num}, stopping...")
                break

            try:
                # Step 1: Run base model to detect all objects
                base_results = base_model.predict(frame, conf=0.6, iou=0.5, max_det=100, verbose=False)
                base_boxes = base_results[0].boxes.xyxy.cpu().numpy() if base_results and len(base_results[0].boxes) > 0 else []
                base_classes = base_results[0].boxes.cls.cpu().numpy() if base_results and len(base_results[0].boxes) > 0 else []
                base_scores = base_results[0].boxes.conf.cpu().numpy() if base_results and len(base_results[0].boxes) > 0 else []

                # Get base model detections with class names
                base_detections = []
                vehicle_detections = []

                for i, box in enumerate(base_boxes):
                    class_name = base_model.names[int(base_classes[i])]
                    detection = {
                        'box': box,
                        'class': class_name,
                        'conf': float(base_scores[i])
                    }
                    base_detections.append(detection)

                    # Track vehicle detections separately
                    if class_name in vehicle_classes:
                        vehicle_detections.append(detection)

                # Step 2: Run waste detection model
                waste_results = waste_model.predict(frame, conf=conf_thresh, iou=0.5, max_det=100, verbose=False)
                waste_boxes = waste_results[0].boxes.xyxy.cpu().numpy() if waste_results and len(waste_results[0].boxes) > 0 else []
                waste_classes = waste_results[0].boxes.cls.cpu().numpy() if waste_results and len(waste_results[0].boxes) > 0 else []
                waste_scores = waste_results[0].boxes.conf.cpu().numpy() if waste_results and len(waste_results[0].boxes) > 0 else []

                # Step 3: Process each waste detection
                frame_detections = []

                for i, waste_box in enumerate(waste_boxes):
                    x1, y1, x2, y2 = waste_box.astype(int)
                    conf = float(waste_scores[i])
                    label = waste_model.names[int(waste_classes[i])]

                    # Check if this waste detection overlaps with non-waste objects
                    is_valid_waste = True
                    for base_det in base_detections:
                        if base_det['class'] in non_waste_classes:
                            iou = calculate_iou(waste_box, base_det['box'])
                            if iou > 0.5:
                                is_valid_waste = False
                                break

                    # Only process valid waste detections
                    if is_valid_waste:
                        waste_center = get_box_center(waste_box)

                        # Check for nearby vehicles within proximity radius
                        nearby_vehicles = []

                        for vehicle_det in vehicle_detections:
                            vehicle_center = get_box_center(vehicle_det['box'])
                            pixel_distance = calculate_pixel_distance(waste_center, vehicle_center)
                            real_distance_ft = pixels_to_feet(pixel_distance, drone_altitude_ft, frame_width)

                            # If vehicle is within proximity radius
                            if real_distance_ft <= proximity_radius_ft:
                                # Try to extract license plate
                                plate_number, plate_conf, plate_detected = extract_plate_number(
                                    frame, vehicle_det['box'], ocr_reader
                                )

                                vehicle_info = {
                                    "vehicle_type": vehicle_det['class'],
                                    "distance_feet": round(real_distance_ft, 2),
                                    "bbox": [int(v) for v in vehicle_det['box']],
                                    "plate_detected": plate_detected
                                }

                                if plate_detected and plate_number:
                                    vehicle_info["plate_number"] = plate_number
                                    vehicle_info["plate_confidence"] = round(plate_conf, 2)

                                nearby_vehicles.append(vehicle_info)

                                # Draw BLUE box around nearby vehicle
                                vx1, vy1, vx2, vy2 = map(int, vehicle_det['box'])
                                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 0), 3)
                                vehicle_label = f"{vehicle_det['class']} ({real_distance_ft:.1f}ft)"
                                if plate_detected and plate_number:
                                    vehicle_label += f" - {plate_number}"
                                cv2.putText(frame, vehicle_label, (vx1, vy1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        # Draw THICK RED bounding box for waste
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
                        waste_label = f"WASTE: {label} {conf:.2f}"
                        if nearby_vehicles:
                            waste_label += f" [{len(nearby_vehicles)} vehicle(s)]"
                        cv2.putText(frame, waste_label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                        # Mock GPS tagging
                        lat, lon = mock_gps(base_lat, base_lon, step, frame_num)

                        # Create detection entry
                        detection_entry = {
                            "class": label,
                            "confidence": conf,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "latitude": lat,
                            "longitude": lon,
                            "nearby_vehicles": nearby_vehicles
                        }

                        frame_detections.append(detection_entry)

                # Draw base model detections (for context, in GREEN)
                for base_det in base_detections:
                    if base_det['class'] in non_waste_classes and base_det['class'] not in vehicle_classes:
                        bx1, by1, bx2, by2 = base_det['box'].astype(int)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 1)
                        cv2.putText(frame, f"{base_det['class']} {base_det['conf']:.2f}",
                                   (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                detections_list.append({
                    "frame": frame_num,
                    "detections": frame_detections
                })

                out.write(frame)

                # Periodic checkpoint
                if (frame_num + 1) % checkpoint_interval == 0:
                    with open(checkpoint_file, "w") as f:
                        json.dump(detections_list, f, indent=2)
                    print(f"\n[CHECKPOINT] Saved progress at frame {frame_num + 1}/{total_frames}")

            except Exception as e:
                print(f"\n[ERROR] Failed to process frame {frame_num}: {e}")
                detections_list.append({
                    "frame": frame_num,
                    "detections": [],
                    "error": str(e)
                })
                continue

    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user")
    finally:
        cap.release()
        out.release()

        # Save final JSON
        with open(output_json, "w") as f:
            json.dump(detections_list, f, indent=2)

        print(f"\n[DONE] Output video saved as: {output_video}")
        print(f"[DONE] Detection JSON saved as: {output_json}")
        print(f"[INFO] Processed {len(detections_list)} frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Drone Trash Detection with Vehicle & Plate Recognition")
    parser.add_argument("--video", required=True, help="Path to input video file (.mp4)")
    parser.add_argument("--lat", type=float, default=24.5230, help="Base latitude for mock GPS")
    parser.add_argument("--lon", type=float, default=54.3820, help="Base longitude for mock GPS")
    parser.add_argument("--step", type=float, default=0.00001, help="GPS increment per frame")
    parser.add_argument("--output_video", default="enhanced_output.mp4", help="Output annotated video")
    parser.add_argument("--output_json", default="enhanced_detections.json", help="Output detection JSON")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold for waste detection. Default: 0.6")
    parser.add_argument("--altitude", type=float, default=100, help="Drone altitude in feet. Default: 100ft")
    parser.add_argument("--radius", type=float, default=15, help="Vehicle proximity radius in feet. Default: 15ft")

    args = parser.parse_args()
    main(args.video, args.lat, args.lon, args.step, args.output_video, args.output_json,
         args.conf, args.altitude, args.radius)
