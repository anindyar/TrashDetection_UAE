import cv2
import argparse
from ultralytics import YOLO

def diagnose_frame(video_path, frame_num):
    """Diagnose what both models detect on a specific frame."""

    # Load both models
    print("[INFO] Loading models...")
    base_model = YOLO("weights/yolov8n.pt")
    waste_model = YOLO("runs/detect/waste_detect/weights/best.pt")

    # Open video and jump to frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        print(f"[ERROR] Could not read frame {frame_num}")
        return

    print(f"\n[INFO] Analyzing frame {frame_num}")
    print("="*60)

    # Run base model with LOW confidence to see everything
    print("\n[BASE MODEL DETECTIONS] (conf >= 0.3):")
    base_results = base_model.predict(frame, conf=0.3, iou=0.5, max_det=100, verbose=False)
    base_boxes = base_results[0].boxes

    if len(base_boxes) > 0:
        for i, box in enumerate(base_boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = base_model.names[cls_id]
            bbox = box.xyxy[0].cpu().numpy()
            print(f"  {i+1}. {class_name}: {conf:.3f} @ [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]")
    else:
        print("  (No detections)")

    # Run waste model with VERY LOW confidence to see everything
    print("\n[WASTE MODEL DETECTIONS] (conf >= 0.3):")
    waste_results = waste_model.predict(frame, conf=0.3, iou=0.5, max_det=100, verbose=False)
    waste_boxes = waste_results[0].boxes

    if len(waste_boxes) > 0:
        for i, box in enumerate(waste_boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = waste_model.names[cls_id]
            bbox = box.xyxy[0].cpu().numpy()
            print(f"  {i+1}. {class_name}: {conf:.3f} @ [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]")
    else:
        print("  (No detections)")

    # Save annotated frame for visual inspection
    annotated_frame = frame.copy()

    # Draw base model detections in GREEN
    if len(base_boxes) > 0:
        for box in base_boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{base_model.names[cls_id]} {conf:.2f}",
                       (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw waste model detections in RED
    if len(waste_boxes) > 0:
        for box in waste_boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
            cv2.putText(annotated_frame, f"WASTE: {waste_model.names[cls_id]} {conf:.2f}",
                       (bbox[0], bbox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    output_file = f"diagnostic_frame_{frame_num}.jpg"
    cv2.imwrite(output_file, annotated_frame)
    print(f"\n[SAVED] Annotated frame saved to: {output_file}")
    print("="*60)

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose detection on specific frames")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--frames", required=True, help="Comma-separated frame numbers (e.g., 100,200,300)")

    args = parser.parse_args()

    frame_numbers = [int(f.strip()) for f in args.frames.split(',')]

    for frame_num in frame_numbers:
        diagnose_frame(args.video, frame_num)
