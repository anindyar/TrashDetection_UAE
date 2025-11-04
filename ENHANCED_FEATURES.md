# Enhanced Trash Detection with Vehicle Proximity & License Plate Recognition

## Overview

This enhanced version adds **vehicle proximity detection** and **license plate recognition** to the drone waste detection system. It can now identify cars within a specified radius of detected trash and attempt to capture their license plate numbers.

## New Features

### 1. Vehicle Proximity Detection
- Detects vehicles (cars, trucks, buses, motorcycles) within configurable radius of trash
- Default radius: **15 feet**
- Calculates real-world distance using drone altitude and camera FOV
- Visual indication: **BLUE boxes** around nearby vehicles

### 2. License Plate Recognition
- Automatic OCR (Optical Character Recognition) on detected vehicles
- Uses EasyOCR with English character support
- Can be extended to Arabic for UAE plates
- Captures plate number with confidence score

### 3. Distance Calculation
- Converts pixel distances to real-world feet
- Uses drone altitude (default: 100ft) and camera FOV (84° for DJI drones)
- Formula: `distance_ft = (pixel_distance / frame_width) * ground_coverage`

### 4. Enhanced Reporting
- **JSON output** with vehicle and plate data
- **Text report** with detailed summary
- **CSV export** for easy data analysis
- Includes: frame number, waste type, GPS location, nearby vehicles, plate numbers

## Installation

### Additional Dependencies

```bash
# Install EasyOCR for license plate recognition
./venv/bin/pip install easyocr
```

**Note:** First run will download OCR language models (~100MB), which may take a few minutes.

## Usage

### Basic Enhanced Detection

```bash
./venv/bin/python enhanced_trash_detect.py \
  --video DJI_20251104123743_0006_D.MP4 \
  --output_video enhanced_output.mp4 \
  --output_json enhanced_detections.json
```

### With Custom Parameters

```bash
./venv/bin/python enhanced_trash_detect.py \
  --video sample.mp4 \
  --altitude 150 \        # Drone altitude in feet
  --radius 20 \           # Vehicle proximity radius in feet
  --conf 0.6 \            # Waste detection confidence threshold
  --output_video output.mp4 \
  --output_json detections.json
```

### Generate Reports

```bash
./venv/bin/python generate_report.py \
  --json enhanced_detections.json \
  --output-txt report.txt \
  --output-csv report.csv
```

## Command-Line Arguments

### enhanced_trash_detect.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video` | str | *required* | Input video file path |
| `--lat` | float | 24.5230 | Base latitude for mock GPS |
| `--lon` | float | 54.3820 | Base longitude for mock GPS |
| `--step` | float | 0.00001 | GPS increment per frame |
| `--output_video` | str | enhanced_output.mp4 | Output annotated video |
| `--output_json` | str | enhanced_detections.json | Output detection JSON |
| `--conf` | float | 0.6 | Waste detection confidence (0.0-1.0) |
| `--altitude` | float | 100 | Drone altitude in feet |
| `--radius` | float | 15 | Vehicle proximity radius in feet |

### generate_report.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--json` | str | *required* | Input detection JSON file |
| `--output-txt` | str | detection_report.txt | Output text report |
| `--output-csv` | str | detection_report.csv | Output CSV report |

## Output Format

### JSON Structure

```json
{
  "frame": 660,
  "detections": [
    {
      "class": "Plastic",
      "confidence": 0.85,
      "bbox": [1700, 1030, 1750, 1090],
      "latitude": 24.5230,
      "longitude": 54.3820,
      "nearby_vehicles": [
        {
          "vehicle_type": "car",
          "distance_feet": 12.34,
          "bbox": [100, 500, 400, 800],
          "plate_detected": true,
          "plate_number": "ABC-1234",
          "plate_confidence": 0.92
        }
      ]
    }
  ]
}
```

### Video Annotations

- **RED boxes (thick, 6px)**: Detected waste
  - Label: `WASTE: [class] [confidence] [[N] vehicle(s)]`
- **BLUE boxes (medium, 3px)**: Nearby vehicles within radius
  - Label: `[vehicle_type] ([distance]ft) - [plate_number]`
- **GREEN boxes (thin, 1px)**: Other objects (context)

## Text Report Example

```
================================================================================
ENHANCED TRASH DETECTION REPORT
================================================================================
Generated: 2025-11-04 15:06:52

VIDEO SUMMARY
--------------------------------------------------------------------------------
Total frames processed: 1515
Frames with waste detected: 45
Total waste detections: 52

WASTE BREAKDOWN
--------------------------------------------------------------------------------
  Plastic: 23
  rubbish: 18
  Metal: 7
  glass: 4

VEHICLE PROXIMITY ANALYSIS
--------------------------------------------------------------------------------
Waste detections with nearby vehicles: 12
Total vehicles detected near waste: 15
License plates captured: 8

LICENSE PLATES CAPTURED
--------------------------------------------------------------------------------
1. Frame 342:
   Plate Number: A12345
   Vehicle Type: car
   Distance from waste: 11.23 feet
   GPS Location: 24.523420, 54.382420

2. Frame 567:
   Plate Number: B67890
   Vehicle Type: truck
   Distance from waste: 14.56 feet
   GPS Location: 24.523670, 54.382670
...
```

## Distance Calculation Details

### Formula

```python
# Ground coverage width at given altitude
ground_coverage_ft = 2 * altitude_ft * tan(FOV_horizontal / 2)

# Feet per pixel
ft_per_pixel = ground_coverage_ft / frame_width

# Real distance
distance_ft = pixel_distance * ft_per_pixel
```

### Assumptions

- **Horizontal FOV**: 84° (standard for DJI drones)
- **Frame Width**: 3840 pixels (4K video)
- **Default Altitude**: 100 feet (~30 meters)
- **Distance Type**: Horizontal (bird's eye view)

### Altitude Adjustment

For accurate distance calculations, adjust `--altitude` based on your drone's flight height:

```bash
# Flying at 150 feet
--altitude 150

# Flying at 50 feet (lower = more accurate plate reading)
--altitude 50
```

**Note:** Lower altitudes provide better license plate resolution but smaller field of view.

## Performance Considerations

### Processing Speed

- **Standard detection**: ~18-20 fps (4K video, CPU)
- **Enhanced detection with OCR**: ~3-5 fps (4K video, CPU)
- **Bottleneck**: OCR processing on vehicle regions

### Optimization Tips

1. **Sample frames**: Process every Nth frame instead of all frames
2. **GPU acceleration**: Use `gpu=True` in EasyOCR reader (requires CUDA)
3. **Lower resolution**: Downscale video before processing
4. **Adjust radius**: Smaller radius = fewer vehicles to process

## Limitations

### License Plate Recognition

1. **Aerial View Challenge**: Plates are small from drone perspective
2. **Angle Dependency**: Best results with top/rear view of vehicles
3. **Resolution**: 4K video recommended; 1080p may struggle
4. **Lighting**: Poor lighting reduces OCR accuracy
5. **Speed**: Slower processing due to OCR overhead

### Distance Accuracy

1. **Altitude Dependency**: Requires accurate altitude data
2. **Terrain**: Assumes flat ground (no elevation changes)
3. **Camera Distortion**: Wide-angle lenses may introduce error at edges
4. **FOV Variation**: Different drone models have different FOVs

## Future Enhancements

- [ ] DJI SRT subtitle parsing for real altitude data
- [ ] Arabic OCR support for UAE plates
- [ ] GPU acceleration option
- [ ] Real-time processing mode
- [ ] Plate region image export for manual verification
- [ ] Integration with vehicle registration database
- [ ] Heat map generation of trash locations
- [ ] Automated email/SMS alerts for detected violations

## Troubleshooting

### OCR Not Detecting Plates

**Possible causes:**
- Altitude too high (plates too small)
- Poor lighting conditions
- Plate not visible from drone angle
- Low video resolution

**Solutions:**
- Reduce altitude (50-100ft recommended)
- Process video in good lighting
- Ensure vehicles visible from above
- Use 4K video

### Inaccurate Distance Calculations

**Possible causes:**
- Wrong altitude parameter
- Terrain elevation changes
- Wrong FOV assumption

**Solutions:**
- Extract real altitude from DJI metadata
- Use `--altitude` flag with actual flight height
- Manual calibration for non-DJI drones

### Slow Processing

**Possible causes:**
- OCR processing on CPU
- High-resolution video (4K)
- Many vehicles in frame

**Solutions:**
- Enable GPU: modify `easyocr.Reader(['en'], gpu=True)`
- Reduce video resolution
- Increase `--radius` threshold to filter distant vehicles

## Technical Details

### Vehicle Classes Detected

- car
- truck
- bus
- motorcycle

### OCR Language Support

Current: English (`en`)

To add Arabic support:
```python
ocr_reader = easyocr.Reader(['en', 'ar'], gpu=False)
```

### Checkpoint System

- Auto-saves progress every 500 frames
- Checkpoint file: `<output_json>.checkpoint`
- Resume-safe (press Ctrl+C safely)

## Examples

### Scenario 1: Plastic Bottle with Nearby Car

**Detection:**
```json
{
  "class": "Plastic",
  "confidence": 0.87,
  "nearby_vehicles": [
    {
      "vehicle_type": "car",
      "distance_feet": 9.45,
      "plate_detected": true,
      "plate_number": "A 45678"
    }
  ]
}
```

**Interpretation:** Plastic bottle detected 9.45 feet from a car with plate "A 45678". Possible littering from vehicle.

### Scenario 2: Trash with No Vehicles

```json
{
  "class": "rubbish",
  "confidence": 0.65,
  "nearby_vehicles": []
}
```

**Interpretation:** General rubbish detected with no vehicles within 15ft radius. Likely existing litter.

## References

- EasyOCR: https://github.com/JaidedAI/EasyOCR
- UAE License Plate Format: [Emirate Code] [1-5 digits]
- DJI Camera Specs: https://www.dji.com/
- YOLO Detection: https://github.com/ultralytics/ultralytics
