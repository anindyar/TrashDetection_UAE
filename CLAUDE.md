# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a drone-based waste/litter detection system using YOLOv8 for computer vision. The system uses a **dual-model approach** to accurately detect waste while avoiding false positives from common objects (cars, people, signs). It processes drone video footage to detect and classify garbage (Metal, Plastic, Glass, and general Rubbish) with mock GPS tagging for each detection.

## Key Architecture

### Dual-Model Detection System

The system uses TWO YOLOv8 models working together:

1. **Base YOLOv8 Model** (COCO dataset - 80 classes)
   - Detects all common objects: person, car, bicycle, traffic light, truck, bus, etc.
   - Preserves original YOLO intelligence
   - Used to filter out non-waste objects
   - Confidence threshold: 0.5

2. **Custom Waste Detection Model** (4 waste classes)
   - Trained on Waste-detect-v-drone-1 dataset (772 aerial images)
   - Detects: Metal, Plastic, glass, rubbish
   - Confidence threshold: 0.75 (adjustable)
   - Training metrics: mAP50 = 0.719, stopped at epoch 14

3. **Smart Filtering via IoU (Intersection over Union)**
   - Calculates overlap between waste detections and normal objects
   - If waste detection overlaps >30% with a car/person/sign → rejected as false positive
   - Only reports waste items that don't overlap with common objects

### Core Components

1. **trash_detect_demo.py** - Main inference script with dual-model detection
   - Loads both base and custom waste models
   - Processes video frame-by-frame with both YOLO models
   - Filters waste detections using IoU-based overlap detection
   - Overlays bounding boxes: GREEN for normal objects, RED for waste
   - Mock GPS tagging system: generates simulated coordinates per frame
   - Error handling: per-frame try/catch to prevent crashes
   - Checkpoint system: saves JSON every 500 frames
   - Outputs annotated video and JSON detection log

2. **train_waste_model.py** - Training script for custom waste model
   - Trains YOLOv8n on Waste-detect-v-drone-1 dataset
   - 50 epochs with early stopping (patience=10)
   - AdamW optimizer, lr=0.00125
   - Batch size 16, image size 640x640
   - Outputs: runs/detect/waste_detect/weights/best.pt

3. **download_roboflow_model.py** - Dataset acquisition
   - Downloads YOLOv8 dataset from Roboflow
   - Workspace: drone-litter-detection
   - Project: waste-detect-v-drone-1val9
   - Requires ROBOFLOW_API_KEY

4. **Waste-detect-v-drone-1/** - Dataset directory
   - 772 annotated aerial drone images
   - Train/valid/test splits with YOLO label files
   - data.yaml defines 4 classes: Metal, Plastic, glass, rubbish
   - CC BY 4.0 license

5. **weights/** - Model weights directory
   - yolov8n.pt: Pretrained base model (COCO dataset)
   - runs/detect/waste_detect/weights/best.pt: Custom trained waste model

### Detection Classes

**Base Model (COCO - 80 classes):**
- person, bicycle, car, motorcycle, airplane, bus, train, truck, boat
- traffic light, fire hydrant, stop sign, parking meter, bench
- And 65+ more common objects

**Waste Model (4 classes):**
- Metal
- Plastic
- glass
- rubbish (general waste)

## Development Commands

### Environment Setup

```bash
# Create virtual environment (Python 3.13)
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install ultralytics roboflow opencv-python torch torchvision numpy tqdm
```

Current installed versions:
- ultralytics==8.3.217
- roboflow==1.2.11
- opencv-python==4.12.0.88
- torch==2.9.0
- numpy==2.2.6

### Running Detection (Dual-Model Approach)

```bash
# Basic usage with trained model (uses default conf=0.75)
./venv/bin/python trash_detect_demo.py --video sample.mp4

# Adjust confidence threshold to reduce false positives (higher = stricter)
./venv/bin/python trash_detect_demo.py --video sample.mp4 --conf 0.85

# Full usage with custom parameters
./venv/bin/python trash_detect_demo.py \
  --video sample.mp4 \
  --lat 24.5230 \
  --lon 54.3820 \
  --step 0.00001 \
  --conf 0.75 \
  --output_video output_waste_detection.mp4 \
  --output_json detection.json

# Outputs:
# - output_waste_detection.mp4: annotated video with GREEN (normal objects) and RED (waste) boxes
# - detection.json: frame-by-frame waste detection log with mock GPS coords
# - detection.json.checkpoint: periodic checkpoint saves (every 500 frames)
```

### Training Custom Model

```bash
# Train using the training script
./venv/bin/python train_waste_model.py

# Or train manually
from ultralytics import YOLO

model = YOLO('weights/yolov8n.pt')
model.train(
    data='Waste-detect-v-drone-1/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='waste_detect',
    patience=10,
    save=True,
    device='cpu'  # or '0' for GPU
)
```

Model saved to: `runs/detect/waste_detect/weights/best.pt`

### Downloading Dataset

```bash
# Using the shell script
./download_model.sh YOUR_ROBOFLOW_API_KEY

# Or directly with Python
export ROBOFLOW_API_KEY=your_key_here
python download_roboflow_model.py

# Or pass API key as argument
python download_roboflow_model.py YOUR_API_KEY
```

Get API key from: https://app.roboflow.com/settings/api

## Technical Details

### Dual-Model Detection Pipeline

1. **Frame Capture**: Read video frame using OpenCV
2. **Base Model Inference**: Detect all objects (cars, people, signs, etc.)
3. **Waste Model Inference**: Detect potential waste items
4. **IoU Filtering**:
   - For each waste detection, check overlap with all base model detections
   - Calculate IoU (Intersection over Union) with non-waste objects
   - If IoU > 0.3 with any car/person/sign → reject as false positive
5. **Visualization**:
   - Draw GREEN boxes for normal objects (context)
   - Draw RED boxes for validated waste detections
6. **Output**: Save only waste detections to JSON, write annotated frame to video

### Error Handling & Recovery

- **Per-frame error handling**: Try/catch around each frame processing
- **Graceful degradation**: Failed frames add empty detection with error message
- **Checkpoint system**: Saves progress every 500 frames to `.checkpoint` file
- **Keyboard interrupt support**: Ctrl+C safely saves partial results
- **Finally block**: Ensures video/JSON files are properly closed and saved

### GPS Mock System

The `mock_gps()` function (trash_detect_demo.py:10-12) generates simulated GPS coordinates:
- Starts at base_lat, base_lon (default: 24.5230, 54.3820)
- Increments by `step` value per frame (default: 0.00001)
- For real drone footage, replace with actual GPS telemetry integration (e.g., from DJI SRT files)

### Output Format

Detection JSON structure:
```json
[
  {
    "frame": 0,
    "detections": [
      {
        "class": "Plastic",
        "confidence": 0.85,
        "bbox": [x1, y1, x2, y2],
        "latitude": 24.5230,
        "longitude": 54.3820
      }
    ]
  },
  {
    "frame": 1,
    "detections": []
  }
]
```

If a frame fails to process, the error is logged:
```json
{
  "frame": 123,
  "detections": [],
  "error": "error message here"
}
```

### Video Processing

- Input: Any OpenCV-compatible video format
- Output codec: mp4v
- Progress tracking: tqdm progress bar
- Processing speed: ~19-20 fps (4K video on CPU with dual models)
- Base model confidence: 0.5
- Waste model confidence: 0.75 (default, adjustable via --conf)
- IoU threshold: 0.5 (NMS), 0.3 (overlap filtering)
- Max detections per frame: 100

## Model Performance

### Custom Waste Detection Model

Trained on 772 aerial drone images from Roboflow dataset:

- **mAP50**: 0.719 (71.9% average precision at 0.5 IoU)
- **mAP50-95**: 0.408
- **Precision**: 0.842
- **Recall**: 0.695
- **Training**: 14 epochs (early stopping activated at epoch 14, patience=10)
- **Dataset split**: 80% train, 15% validation, 5% test
- **Best model**: runs/detect/waste_detect/weights/best.pt

Training results available in: `runs/detect/waste_detect/`
- results.csv: epoch-by-epoch metrics
- confusion_matrix.png: class confusion analysis
- val_batch*_pred.jpg: validation predictions

## Troubleshooting

### False Positives / Detecting Everything as Rubbish

**Problem**: Model detects non-waste objects (cars, road signs, people) as waste

**Root Cause**: Original single-model approach only knew 4 waste classes, forced everything into those categories

**Solution**: ✅ **Now FIXED with dual-model approach**

The new system:
1. Uses base YOLO to detect all normal objects (cars, people, signs)
2. Uses waste model to detect waste
3. Filters out waste detections that overlap with normal objects
4. Only flags actual waste items

If you still see false positives:
- Increase confidence threshold: `--conf 0.80` or `--conf 0.85`
- Check the video is drone/aerial footage (model trained on aerial perspective)
- Review GREEN boxes in output video to see what normal objects were detected

### Domain Mismatch

**Training data**: Aerial drone footage (bird's eye view) of waste on the ground

**Best results with**:
- Drone/aerial footage
- Bird's eye view perspective
- Similar altitude and angle to training data

**May struggle with**:
- Ground-level or road-scene videos
- Side-view or angled perspectives
- Indoor scenes

**Solution**: For different perspectives, retrain with mixed dataset including those viewpoints

### Processing Crashes / Memory Issues

**Symptoms**: Process crashes partway through video, exit code 1

**Causes**:
- 4K video processing uses significant memory
- Corrupted video frames
- Dual model inference is memory-intensive

**Solutions**:
✅ **Now mitigated with error handling + checkpointing**

- Per-frame error handling prevents single bad frame from crashing entire process
- Checkpoint saves every 500 frames preserve progress
- If crash occurs, check `detection.json.checkpoint` for partial results
- For very large videos, consider processing in chunks or reducing resolution

### Checkpoint Recovery

If process is interrupted:
```bash
# Checkpoint file contains progress up to last 500-frame boundary
ls -lh detection.json.checkpoint

# Copy checkpoint to main output
cp detection.json.checkpoint detection.json
```

## File Locations

- Input videos: Place in project root (e.g., sample.mp4)
- Base model weights: weights/yolov8n.pt (COCO pretrained)
- Custom model weights: runs/detect/waste_detect/weights/best.pt
- Dataset: Waste-detect-v-drone-1/ (downloaded via Roboflow)
- Training outputs: runs/detect/waste_detect/
- Detection outputs: Generated in project root by default
- Checkpoints: `<output_json>.checkpoint` (e.g., detection.json.checkpoint)

## Project Status

✅ Dataset downloaded (772 images, 4 classes)
✅ Custom waste model trained (mAP50: 0.719)
✅ Dual-model detection system implemented
✅ Error handling and checkpointing added
✅ IoU-based filtering to reduce false positives
✅ Ready for production use on aerial drone footage

## References

- Roboflow Dataset: https://universe.roboflow.com/drone-litter-detection/waste-detect-v-drone-1val9/dataset/1
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- COCO Dataset Classes: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
