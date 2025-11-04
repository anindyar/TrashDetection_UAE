import cv2
import json
import argparse
from ultralytics import YOLO
from tqdm import tqdm

# -------------------------------
# Helper function to mock GPS data
# -------------------------------
def mock_gps(lat, lon, step, frame_num):
    """Generate mock GPS coordinates per frame."""
    return lat + step * frame_num, lon + step * frame_num

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
# Main Function
# -------------------------------
def main(video_path, base_lat, base_lon, step, output_video, output_json, conf_thresh=0.75):
    # Load BOTH models
    print("[INFO] Loading YOLOv8 models...")
    print(f"[INFO] Using confidence threshold: {conf_thresh}")

    # Base model - detects everything (person, car, traffic light, etc.)
    print("[INFO] Loading base YOLOv8 model (COCO dataset - 80 classes)...")
    base_model = YOLO("weights/yolov8n.pt")

    # Waste detection model - detects waste items
    print("[INFO] Loading custom waste detection model (4 waste classes)...")
    waste_model = YOLO("runs/detect/waste_detect/weights/best.pt")

    # Define non-waste classes from COCO that should NOT be flagged as trash
    # These are common objects we expect to see in the scene
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

    # Checkpoint interval (save JSON every N frames to prevent data loss)
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
                # Step 1: Run base model to detect all objects (cars, people, signs, etc.)
                # Using higher confidence (0.6) to reduce false detections from base model
                base_results = base_model.predict(frame, conf=0.6, iou=0.5, max_det=100, verbose=False)
                base_boxes = base_results[0].boxes.xyxy.cpu().numpy() if base_results and len(base_results[0].boxes) > 0 else []
                base_classes = base_results[0].boxes.cls.cpu().numpy() if base_results and len(base_results[0].boxes) > 0 else []
                base_scores = base_results[0].boxes.conf.cpu().numpy() if base_results and len(base_results[0].boxes) > 0 else []

                # Get base model detections with class names
                base_detections = []
                for i, box in enumerate(base_boxes):
                    class_name = base_model.names[int(base_classes[i])]
                    base_detections.append({
                        'box': box,
                        'class': class_name,
                        'conf': float(base_scores[i])
                    })

                # Step 2: Run waste detection model
                waste_results = waste_model.predict(frame, conf=conf_thresh, iou=0.5, max_det=100, verbose=False)
                waste_boxes = waste_results[0].boxes.xyxy.cpu().numpy() if waste_results and len(waste_results[0].boxes) > 0 else []
                waste_classes = waste_results[0].boxes.cls.cpu().numpy() if waste_results and len(waste_results[0].boxes) > 0 else []
                waste_scores = waste_results[0].boxes.conf.cpu().numpy() if waste_results and len(waste_results[0].boxes) > 0 else []

                # Step 3: Filter waste detections - remove those that overlap with non-waste objects
                frame_detections = []
                for i, waste_box in enumerate(waste_boxes):
                    x1, y1, x2, y2 = waste_box.astype(int)
                    conf = float(waste_scores[i])
                    label = waste_model.names[int(waste_classes[i])]

                    # Check if this waste detection overlaps significantly with any non-waste object
                    is_valid_waste = True
                    overlapping_object = None

                    for base_det in base_detections:
                        if base_det['class'] in non_waste_classes:
                            iou = calculate_iou(waste_box, base_det['box'])
                            # If IoU > 0.5 (50% overlap), it's likely part of a normal object, not waste
                            # Increased threshold to 0.5 to reduce false rejections of actual waste
                            if iou > 0.5:
                                is_valid_waste = False
                                overlapping_object = base_det['class']
                                break

                    # Only add to detections if it's valid waste (not overlapping with common objects)
                    if is_valid_waste:
                        # Draw THICK RED bounding box for waste (increased from 2 to 6 for visibility)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
                        cv2.putText(frame, f"WASTE: {label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                        # Mock GPS tagging
                        lat, lon = mock_gps(base_lat, base_lon, step, frame_num)
                        frame_detections.append({
                            "class": label,
                            "confidence": conf,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "latitude": lat,
                            "longitude": lon
                        })

                # Step 4: Draw base model detections (for context, in GREEN)
                for base_det in base_detections:
                    if base_det['class'] in non_waste_classes:
                        bx1, by1, bx2, by2 = base_det['box'].astype(int)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 1)
                        cv2.putText(frame, f"{base_det['class']} {base_det['conf']:.2f}",
                                    (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                detections_list.append({
                    "frame": frame_num,
                    "detections": frame_detections
                })

                out.write(frame)

                # Periodic checkpoint to prevent data loss
                if (frame_num + 1) % checkpoint_interval == 0:
                    with open(checkpoint_file, "w") as f:
                        json.dump(detections_list, f, indent=2)
                    print(f"\n[CHECKPOINT] Saved progress at frame {frame_num + 1}/{total_frames}")

            except Exception as e:
                print(f"\n[ERROR] Failed to process frame {frame_num}: {e}")
                print(f"[INFO] Continuing with next frame...")
                # Add empty detection for failed frame
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
    parser = argparse.ArgumentParser(description="Drone Trash Detection Demo")
    parser.add_argument("--video", required=True, help="Path to input video file (.mp4)")
    parser.add_argument("--lat", type=float, default=24.5230, help="Base latitude for mock GPS")
    parser.add_argument("--lon", type=float, default=54.3820, help="Base longitude for mock GPS")
    parser.add_argument("--step", type=float, default=0.00001, help="GPS increment per frame")
    parser.add_argument("--output_video", default="output.mp4", help="Output annotated video")
    parser.add_argument("--output_json", default="detections.json", help="Output detection JSON")
    parser.add_argument("--conf", type=float, default=0.75, help="Confidence threshold (0.0-1.0). Higher = fewer false positives. Default: 0.75")

    args = parser.parse_args()
    main(args.video, args.lat, args.lon, args.step, args.output_video, args.output_json, args.conf)

