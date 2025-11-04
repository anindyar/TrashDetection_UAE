# Drone Waste Detection - Complete Process Documentation

This document provides a comprehensive, step-by-step explanation of the entire waste detection process, from setup to final output.

## Table of Contents

1. [System Overview](#system-overview)
2. [Setup and Installation](#setup-and-installation)
3. [Dataset Acquisition](#dataset-acquisition)
4. [Model Training Process](#model-training-process)
5. [Detection Pipeline](#detection-pipeline)
6. [Technical Deep Dive](#technical-deep-dive)
7. [Output Analysis](#output-analysis)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## System Overview

### Purpose

This system detects waste and litter in drone/aerial video footage, providing GPS-tagged detections for environmental monitoring, cleanup operations, and waste mapping applications.

### Architecture Summary

The system uses a **dual-model approach** combining:

1. **Base YOLOv8 Model** - Pre-trained on COCO dataset (80 object classes)
   - Purpose: Detect normal objects (cars, people, signs, buildings, etc.)
   - Prevents false positives by identifying what is NOT waste

2. **Custom Waste Model** - Trained on aerial waste dataset (4 waste classes)
   - Purpose: Detect waste items (Metal, Plastic, glass, rubbish)
   - Specialized for aerial/drone perspective

3. **IoU-Based Filter** - Intersection over Union overlap detection
   - Purpose: Remove waste detections that overlap with normal objects
   - Result: Only actual waste items are flagged

### Key Features

- ✅ Dual-model intelligence to reduce false positives
- ✅ Real-time progress tracking with visual output
- ✅ GPS tagging for each detection (mock or real GPS)
- ✅ Error handling and checkpoint system for reliability
- ✅ Annotated video output with color-coded bounding boxes
- ✅ Structured JSON output for downstream processing

---

## Setup and Installation

### Prerequisites

- Python 3.13 (or 3.8+)
- ~5GB disk space for models and dependencies
- CPU or NVIDIA GPU (optional, but faster)

### Step-by-Step Installation

```bash
# 1. Navigate to project directory
cd /home/ar/em-csr

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Install required packages
pip install ultralytics roboflow opencv-python torch torchvision numpy tqdm

# 5. Verify installation
python -c "from ultralytics import YOLO; print('YOLO installed successfully')"
```

### Dependency Versions

The system has been tested with:
- ultralytics==8.3.217
- roboflow==1.2.11
- opencv-python==4.12.0.88
- torch==2.9.0
- numpy==2.2.6
- tqdm (latest)

---

## Dataset Acquisition

### Overview

The waste detection model is trained on the **Waste-detect-v-drone-1** dataset from Roboflow, which contains 772 annotated aerial images of waste items.

### Dataset Details

- **Source**: Roboflow Universe
- **Project**: drone-litter-detection/waste-detect-v-drone-1val9
- **Images**: 772 total (80% train, 15% validation, 5% test)
- **Perspective**: Aerial/drone bird's eye view
- **License**: CC BY 4.0
- **Format**: YOLOv8 annotations

### Classes (4 total)

1. **Metal** - Metal waste items (cans, scrap metal, foil)
2. **Plastic** - Plastic bottles, bags, containers
3. **glass** - Glass bottles, broken glass
4. **rubbish** - General waste, mixed materials

### Download Process

```bash
# Option 1: Using the download script
./download_model.sh YOUR_ROBOFLOW_API_KEY

# Option 2: Using Python script with environment variable
export ROBOFLOW_API_KEY=your_api_key_here
python download_roboflow_model.py

# Option 3: Using Python script with argument
python download_roboflow_model.py YOUR_API_KEY
```

**Get your API key**: https://app.roboflow.com/settings/api

### Dataset Structure

After download, the directory structure is:

```
Waste-detect-v-drone-1/
├── data.yaml           # Dataset configuration (classes, paths)
├── README.roboflow.txt # Dataset information
├── train/
│   ├── images/        # Training images
│   └── labels/        # YOLO format annotations (.txt)
├── valid/
│   ├── images/        # Validation images
│   └── labels/        # Validation annotations
└── test/
    ├── images/        # Test images
    └── labels/        # Test annotations
```

---

## Model Training Process

### Overview

Training creates a custom YOLOv8 model specialized in detecting waste from aerial footage.

### Training Configuration

**File**: `train_waste_model.py`

Key parameters:
- **Base model**: YOLOv8n (nano - fastest, smallest)
- **Epochs**: 50 maximum
- **Early stopping**: patience=10 (stops if no improvement for 10 epochs)
- **Image size**: 640x640 pixels
- **Batch size**: 16
- **Device**: CPU (change to '0' for GPU)
- **Optimizer**: AdamW (automatic)
- **Learning rate**: 0.00125 (automatic)

### Training Command

```bash
# Using the training script
./venv/bin/python train_waste_model.py
```

### Training Process Flow

1. **Initialization**
   - Load pretrained YOLOv8n weights (trained on COCO dataset)
   - Initialize model architecture

2. **Data Loading**
   - Read data.yaml configuration
   - Load training images and annotations
   - Apply augmentation (automatic by Ultralytics)

3. **Training Loop** (up to 50 epochs)
   - Forward pass: predict bounding boxes and classes
   - Calculate loss: box loss + class loss + DFL loss
   - Backward pass: update weights
   - Validate on validation set
   - Save best model when validation mAP improves

4. **Early Stopping**
   - Monitor validation mAP50 (mean Average Precision at 0.5 IoU)
   - If no improvement for 10 epochs → stop training
   - Prevents overfitting

5. **Output**
   - Best model: `runs/detect/waste_detect/weights/best.pt`
   - Last model: `runs/detect/waste_detect/weights/last.pt`
   - Training metrics: `runs/detect/waste_detect/results.csv`
   - Visualizations: confusion matrix, PR curves, validation predictions

### Training Results

**Our trained model achieved**:

| Metric | Value | Meaning |
|--------|-------|---------|
| **mAP50** | 0.719 | 71.9% accuracy at 0.5 IoU threshold |
| **mAP50-95** | 0.408 | 40.8% accuracy averaged across IoU 0.5-0.95 |
| **Precision** | 0.842 | 84.2% of detections are correct |
| **Recall** | 0.695 | 69.5% of waste items are detected |
| **Epochs** | 14 | Training stopped early (no improvement after epoch 4) |

**Interpretation**:
- High precision (84%) → Low false positives (few non-waste items flagged as waste)
- Good recall (70%) → Detects most waste items, but misses ~30%
- mAP50 of 72% is good for a small dataset (772 images)

### Training Time

- **CPU**: ~30-45 minutes for 14 epochs
- **GPU (NVIDIA)**: ~10-15 minutes for 14 epochs

---

## Detection Pipeline

### Overview

The detection pipeline processes video footage frame-by-frame using dual-model inference and smart filtering.

### Pipeline Architecture

```
Video Input (MP4/AVI/etc.)
         ↓
    Frame Extraction (OpenCV)
         ↓
    ┌────────────────────────┐
    │  PARALLEL INFERENCE    │
    ├────────────────────────┤
    │ 1. Base YOLO Model     │ → Detects: cars, people, signs, etc. (80 classes)
    │ 2. Waste YOLO Model    │ → Detects: Metal, Plastic, glass, rubbish
    └────────────────────────┘
         ↓
    IoU Filtering
    (Remove waste detections overlapping with normal objects)
         ↓
    Visualization
    (Draw GREEN boxes for normal objects, RED boxes for waste)
         ↓
    Output: Annotated Video + JSON Detections
```

### Step-by-Step Process

#### 1. Initialization

```python
# Load both models
base_model = YOLO("weights/yolov8n.pt")  # COCO 80 classes
waste_model = YOLO("runs/detect/waste_detect/weights/best.pt")  # 4 waste classes
```

#### 2. Video Setup

```python
# Open input video
cap = cv2.VideoCapture(video_path)

# Get video properties
width = 3840  # 4K
height = 2160
fps = 30.0
total_frames = 20158

# Create output video writer
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
```

#### 3. Frame-by-Frame Processing

For each frame (0 to 20157):

##### a. Base Model Inference

```python
# Detect all objects (cars, people, signs, etc.)
base_results = base_model.predict(frame, conf=0.5)
base_detections = [
    {'box': [x1, y1, x2, y2], 'class': 'car', 'conf': 0.85},
    {'box': [x1, y1, x2, y2], 'class': 'person', 'conf': 0.92},
    ...
]
```

##### b. Waste Model Inference

```python
# Detect waste items
waste_results = waste_model.predict(frame, conf=0.75)
waste_detections = [
    {'box': [x1, y1, x2, y2], 'class': 'Plastic', 'conf': 0.88},
    {'box': [x1, y1, x2, y2], 'class': 'rubbish', 'conf': 0.76},
    ...
]
```

##### c. IoU Filtering

For each waste detection:

```python
for waste in waste_detections:
    is_valid_waste = True

    # Check overlap with all normal objects
    for base_obj in base_detections:
        if base_obj['class'] in non_waste_classes:  # car, person, sign, etc.
            iou = calculate_iou(waste['box'], base_obj['box'])

            if iou > 0.3:  # >30% overlap
                is_valid_waste = False  # Likely part of normal object
                break

    if is_valid_waste:
        # This is real waste - add to output
        frame_detections.append(waste)
```

**IoU Calculation**:

```
IoU = Area of Intersection / Area of Union

Example:
Waste box: [100, 100, 200, 200]  (100x100 pixels)
Car box:   [150, 150, 300, 300]  (150x150 pixels)

Intersection: [150, 150, 200, 200] = 50x50 = 2500 pixels
Union: 10000 + 22500 - 2500 = 30000 pixels
IoU = 2500 / 30000 = 0.083 (8.3% overlap)

→ Since 0.083 < 0.3, this waste detection is VALID
```

##### d. Visualization

```python
# Draw GREEN boxes for normal objects (context)
for base_det in base_detections:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green
    cv2.putText(frame, f"{class_name} {conf:.2f}", ...)

# Draw RED boxes for validated waste
for waste in validated_waste:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
    cv2.putText(frame, f"WASTE: {class_name} {conf:.2f}", ...)
```

##### e. GPS Tagging

```python
# Mock GPS (for testing)
lat, lon = mock_gps(base_lat=24.5230, base_lon=54.3820, step=0.00001, frame_num)

# Real GPS (for production)
# Read from drone telemetry (DJI SRT files, etc.)
lat, lon = read_gps_from_telemetry(frame_num)
```

##### f. Save Detection Data

```python
detections_list.append({
    "frame": frame_num,
    "detections": [
        {
            "class": "Plastic",
            "confidence": 0.88,
            "bbox": [100, 150, 200, 250],
            "latitude": 24.5230,
            "longitude": 54.3820
        }
    ]
})
```

##### g. Write Frame to Output Video

```python
out.write(frame)  # Annotated frame with boxes
```

##### h. Checkpoint (every 500 frames)

```python
if (frame_num + 1) % 500 == 0:
    # Save intermediate results
    with open("detection.json.checkpoint", "w") as f:
        json.dump(detections_list, f, indent=2)
    print(f"[CHECKPOINT] Saved progress at frame {frame_num + 1}/{total_frames}")
```

#### 4. Finalization

```python
# Close video files
cap.release()
out.release()

# Save final JSON
with open("detection.json", "w") as f:
    json.dump(detections_list, f, indent=2)

print("[DONE] Processing complete!")
```

### Detection Command

```bash
# Basic usage
./venv/bin/python trash_detect_demo.py --video sample.mp4

# Full usage with all parameters
./venv/bin/python trash_detect_demo.py \
  --video sample.mp4 \
  --conf 0.75 \
  --output_video output_waste_detection.mp4 \
  --output_json detection.json \
  --lat 24.5230 \
  --lon 54.3820 \
  --step 0.00001
```

### Performance Metrics

**Processing Speed** (4K video, CPU):
- Dual-model approach: ~19-20 fps
- Total time for 11-minute video (20,158 frames): ~17-18 minutes

**Memory Usage**:
- Peak RAM: ~2-3 GB
- GPU VRAM (if using GPU): ~1-2 GB

---

## Technical Deep Dive

### IoU (Intersection over Union) Explained

IoU measures the overlap between two bounding boxes.

**Formula**:
```
IoU = Area of Intersection / Area of Union
```

**Visual Example**:

```
Box A (Waste):        Box B (Car):
┌─────┐
│     │              ┌──────────┐
│  A  │──────────┐   │          │
│     │    ▓▓▓▓  │   │    B     │
└─────┴────▓▓▓▓──┘   │          │
           └──────────┘

▓▓▓▓ = Intersection
Total shaded area = Union
```

**Our threshold**: IoU > 0.3 means "likely the same object"

### Non-Waste Object Classes

The base model can detect 80 COCO classes. We filter out these as non-waste:

**Vehicles**: car, motorcycle, airplane, bus, train, truck, boat, bicycle

**People & Animals**: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Traffic**: traffic light, fire hydrant, stop sign, parking meter

**Furniture**: chair, couch, bed, dining table, toilet, bench

**Electronics**: tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, refrigerator

**Accessories**: backpack, umbrella, handbag, tie, suitcase

**Sports**: frisbee, skis, snowboard, sports ball, kite, baseball bat, skateboard, surfboard, tennis racket

**Household**: book, clock, vase, scissors, teddy bear, hair drier, toothbrush, potted plant, sink

### Error Handling Strategy

**Problem**: Video processing can fail due to corrupted frames, memory issues, or codec problems

**Solution**: Multi-layer error handling

1. **Per-frame try/catch**:
   ```python
   try:
       # Process frame
   except Exception as e:
       # Log error, add empty detection, continue
       detections_list.append({"frame": N, "detections": [], "error": str(e)})
   ```

2. **Keyboard interrupt handling**:
   ```python
   try:
       # Main loop
   except KeyboardInterrupt:
       print("User interrupted - saving partial results")
   ```

3. **Finally block cleanup**:
   ```python
   finally:
       cap.release()
       out.release()
       save_json(detections_list)
   ```

4. **Checkpoint system**: Saves every 500 frames

Result: **No data loss**, even if process crashes or is interrupted

### Confidence Threshold Tuning

**Trade-off**: Precision vs Recall

```
Low threshold (0.5):
  ✅ High recall (detects more waste)
  ❌ More false positives (flags non-waste as waste)

High threshold (0.9):
  ✅ High precision (few false positives)
  ❌ Lower recall (misses some real waste)

Recommended (0.75):
  ⚖️ Balanced approach
```

**Adjustment guide**:

```bash
# Too many false positives? Increase threshold
--conf 0.80  # or 0.85, 0.90

# Missing real waste? Decrease threshold
--conf 0.70  # or 0.65, 0.60
```

---

## Output Analysis

### Video Output

**File**: `output_waste_detection.mp4`

**Visual elements**:
- **GREEN boxes** (thin, 1px): Normal objects detected by base model
  - Example: "car 0.92", "person 0.85"
  - Purpose: Show what was filtered out, provide context

- **RED boxes** (thick, 2px): Waste items detected and validated
  - Example: "WASTE: Plastic 0.88", "WASTE: rubbish 0.76"
  - Purpose: Highlight actual waste for attention

**Video properties**:
- Resolution: Same as input (e.g., 3840x2160 for 4K)
- FPS: Same as input (e.g., 30 fps)
- Codec: mp4v (H.264 compatible)

### JSON Output

**File**: `detection.json`

**Structure**:

```json
[
  {
    "frame": 0,
    "detections": []
  },
  {
    "frame": 54,
    "detections": [
      {
        "class": "Plastic",
        "confidence": 0.88,
        "bbox": [1250, 890, 1350, 990],
        "latitude": 24.52354,
        "longitude": 54.38254
      },
      {
        "class": "rubbish",
        "confidence": 0.76,
        "bbox": [2100, 1200, 2250, 1350],
        "latitude": 24.52354,
        "longitude": 54.38254
      }
    ]
  },
  {
    "frame": 55,
    "detections": []
  }
]
```

**Fields explained**:

- `frame`: Frame number (0-indexed)
- `detections`: Array of waste items found in this frame
  - `class`: Waste type (Metal, Plastic, glass, rubbish)
  - `confidence`: Model confidence score (0.0-1.0)
  - `bbox`: Bounding box [x1, y1, x2, y2] in pixels
    - (x1, y1): Top-left corner
    - (x2, y2): Bottom-right corner
  - `latitude`, `longitude`: GPS coordinates (mock or real)

**Error logging**:

If a frame fails to process:
```json
{
  "frame": 1795,
  "detections": [],
  "error": "CUDA out of memory"
}
```

### Checkpoint Output

**File**: `detection.json.checkpoint`

- Automatically saved every 500 frames
- Contains same format as main JSON
- Used for crash recovery

**Recovery**:
```bash
# If process crashes, recover from checkpoint
cp detection.json.checkpoint detection.json
```

---

## Troubleshooting Guide

### Issue 1: False Positives (Cars/Signs Detected as Waste)

**Symptoms**: Output video shows RED boxes on cars, road signs, people

**Diagnosis**:
```bash
# Check if dual-model system is active
grep "Loading base YOLOv8 model" trash_detect_demo.py  # Should exist

# Review first few detections
head -100 detection.json  # Should have empty frames initially
```

**Solutions**:

1. **Increase confidence threshold**:
   ```bash
   --conf 0.85  # Try higher values: 0.80, 0.85, 0.90
   ```

2. **Verify dual-model is running**:
   - Check console output for "Loading base YOLOv8 model"
   - Should see both models loading at start

3. **Check video perspective**:
   - Model trained on aerial footage
   - Works best with bird's eye view
   - May struggle with ground-level videos

### Issue 2: Processing Crashes

**Symptoms**: Script exits with error code 1, incomplete outputs

**Diagnosis**:
```bash
# Check available memory
free -h

# Check disk space
df -h .

# Check if checkpoint exists
ls -lh detection.json.checkpoint
```

**Solutions**:

1. **Recover from checkpoint**:
   ```bash
   cp detection.json.checkpoint detection.json
   # Contains progress up to last 500-frame boundary
   ```

2. **Reduce memory usage**:
   ```bash
   # Process smaller video segments
   ffmpeg -i sample.mp4 -ss 00:00:00 -t 00:05:00 segment1.mp4
   ```

3. **Check for corrupted frames**:
   ```bash
   # Validate video
   ffmpeg -v error -i sample.mp4 -f null -
   ```

### Issue 3: Slow Processing

**Symptoms**: Very slow FPS (~5 fps instead of ~20 fps)

**Diagnosis**:
```bash
# Check CPU usage
top  # Should be near 100% on one core

# Check if GPU is available but not used
nvidia-smi  # If NVIDIA GPU present
```

**Solutions**:

1. **Use GPU acceleration**:
   ```python
   # Edit train_waste_model.py or trash_detect_demo.py
   device='0'  # Instead of 'cpu'
   ```

2. **Reduce video resolution**:
   ```bash
   # Downscale 4K to 1080p
   ffmpeg -i sample.mp4 -vf scale=1920:1080 sample_1080p.mp4
   ```

3. **Use smaller model** (already using smallest - YOLOv8n)

### Issue 4: Missing Real Waste

**Symptoms**: Known waste items not detected (high false negatives)

**Diagnosis**:
```bash
# Check confidence threshold
grep "conf_thresh" trash_detect_demo.py

# Check model performance
cat runs/detect/waste_detect/results.csv | tail -5
```

**Solutions**:

1. **Lower confidence threshold**:
   ```bash
   --conf 0.65  # Try: 0.70, 0.65, 0.60
   ```

2. **Check video perspective**:
   - Ensure aerial/drone footage
   - Similar angle to training data

3. **Retrain with more data**:
   - Add images of missed waste types
   - Augment dataset with similar scenarios

### Issue 5: JSON Parsing Errors

**Symptoms**: Cannot read detection.json in other programs

**Diagnosis**:
```bash
# Validate JSON syntax
python -m json.tool detection.json > /dev/null

# Check file size
ls -lh detection.json
```

**Solutions**:

1. **Check file is complete**:
   ```bash
   tail detection.json  # Should end with ]
   ```

2. **Use checkpoint if main file corrupted**:
   ```bash
   python -m json.tool detection.json.checkpoint > detection.json
   ```

---

## Advanced Usage

### Real GPS Integration

Replace mock GPS with actual drone telemetry:

```python
# Instead of mock_gps()
def read_gps_from_telemetry(frame_num, srt_file):
    """Read GPS from DJI SRT file"""
    # Parse SRT subtitle file (DJI drone telemetry)
    # Match frame number to timestamp
    # Extract GPS coordinates
    return lat, lon
```

### Custom Waste Classes

To add new waste classes (e.g., "batteries", "electronics"):

1. **Collect images** of new waste types from aerial perspective
2. **Annotate** using Roboflow or LabelImg
3. **Update data.yaml**:
   ```yaml
   names:
   - Metal
   - Plastic
   - glass
   - rubbish
   - batteries  # New class
   - electronics  # New class
   nc: 6
   ```
4. **Retrain model** with expanded dataset

### Batch Processing

Process multiple videos:

```bash
for video in videos/*.mp4; do
    ./venv/bin/python trash_detect_demo.py \
        --video "$video" \
        --output_video "output/${video%.mp4}_detected.mp4" \
        --output_json "output/${video%.mp4}_detections.json"
done
```

### Real-Time Detection

For live drone feed (RTSP stream):

```python
# Modify trash_detect_demo.py
cap = cv2.VideoCapture("rtsp://drone-ip:554/stream")
# Process frames in real-time
# Display with cv2.imshow() for live preview
```

---

## Conclusion

This waste detection system provides a robust, production-ready solution for identifying litter in aerial drone footage. The dual-model approach ensures high accuracy while minimizing false positives, making it suitable for real-world environmental monitoring applications.

**Key Takeaways**:

✅ **Accurate**: 72% mAP50, 84% precision
✅ **Robust**: Error handling, checkpointing, crash recovery
✅ **Flexible**: Adjustable confidence, GPU/CPU support
✅ **Production-ready**: GPS tagging, structured output, batch processing

For questions or issues, refer to CLAUDE.md or open an issue on the project repository.
