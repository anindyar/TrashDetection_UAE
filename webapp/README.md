# 🗑️ Drone Trash Detection Web Application

Interactive web interface for detecting and mapping trash from DJI drone footage with vehicle proximity tracking.

## Features

- **📁 Drag & Drop Upload**: Easy file upload for DJI video (.mp4) and subtitle (.srt) files
- **🗺️ Interactive Map**: View all detected trash locations on OpenStreetMap
- **📸 Frame Preview**: Click markers to see actual frame with detected trash
- **🚗 Vehicle Tracking**: See nearby vehicles within proximity radius
- **🔢 License Plate Detection**: Automatic OCR for vehicle identification
- **📊 Statistics Dashboard**: Real-time stats on detections and vehicles
- **⚙️ Configurable**: Adjust confidence threshold and proximity radius

## Quick Start

### 1. Start the Web Server

```bash
cd webapp
./start_webapp.sh
```

### 2. Access the Interface

Open your browser and navigate to:
```
http://localhost:5000
```

### 3. Upload Files

1. Drag & drop your DJI video (.mp4) and subtitle (.srt) files
2. Or click the upload area to browse for files
3. Adjust settings if needed:
   - **Confidence Threshold**: 0.3-0.9 (default: 0.6)
   - **Proximity Radius**: 5-50 feet (default: 15)
4. Click "Process Video"

### 4. View Results

Once processing completes:
- **Statistics** show total detections, vehicles, and plates
- **Interactive Map** displays all trash locations
- **Click any marker** to see:
  - Frame image with trash detection
  - Trash type and confidence
  - GPS coordinates
  - Nearby vehicles (if any)
  - License plate numbers (if detected)

## Architecture

```
webapp/
├── app.py              # Flask backend API
├── templates/
│   └── index.html      # Frontend interface
├── uploads/            # Temporary upload storage
├── results/            # Processing results
│   └── <job-id>/
│       ├── input_video.mp4
│       ├── input_video.srt
│       ├── detections.json
│       └── output_video.mp4
└── start_webapp.sh     # Startup script
```

## API Endpoints

### `POST /upload`
Upload video and SRT files.

**Request**: multipart/form-data with `video` and `srt` files

**Response**:
```json
{
  "job_id": "uuid",
  "message": "Files uploaded successfully"
}
```

### `POST /process/<job_id>`
Start processing uploaded files.

**Request Body**:
```json
{
  "conf": 0.6,
  "radius": 15
}
```

**Response**:
```json
{
  "job_id": "uuid",
  "status": "processing"
}
```

### `GET /status/<job_id>`
Check processing status.

**Response**:
```json
{
  "status": "processing|complete|unknown",
  "job_id": "uuid"
}
```

### `GET /results/<job_id>`
Get detection results with map data.

**Response**:
```json
{
  "job_id": "uuid",
  "total_detections": 26,
  "markers": [
    {
      "frame": 686,
      "lat": 24.835647,
      "lon": 55.374196,
      "class": "rubbish",
      "confidence": 0.607,
      "nearby_vehicles": [
        {
          "vehicle_type": "car",
          "distance_feet": 8.22,
          "plate_detected": false
        }
      ]
    }
  ]
}
```

### `GET /frame/<job_id>/<frame_num>`
Get specific frame as base64 image.

**Response**:
```json
{
  "frame": 686,
  "image": "data:image/jpeg;base64,..."
}
```

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5 + CSS3 + JavaScript
- **Map**: Leaflet.js + OpenStreetMap
- **Detection**: YOLOv8 + EasyOCR
- **Processing**: OpenCV + PyTorch

## Configuration

Edit settings in the web interface:

- **Confidence Threshold** (0.3-0.9): Higher = fewer false positives
- **Proximity Radius** (5-50 ft): Distance to check for nearby vehicles

## Performance

- **Upload**: Instant (files saved locally)
- **Processing**: ~2 minutes for 25-second 4K video (CPU mode)
- **Results**: Instant display on interactive map

## Troubleshooting

### Web server won't start
```bash
# Check if port 5000 is already in use
lsof -i :5000

# Or use a different port
cd webapp
../venv/bin/python app.py --port 8080
```

### Processing stuck
- Check `webapp/results/<job_id>/` for error logs
- Ensure sufficient disk space for video processing
- Check that venv is properly set up

### Map not showing
- Ensure internet connection (for OpenStreetMap tiles)
- Check browser console for JavaScript errors
- Verify detection results have valid GPS coordinates

## Future Enhancements

- [ ] Real-time progress bar during processing
- [ ] Export results as CSV/GeoJSON
- [ ] Multi-video batch processing
- [ ] User authentication
- [ ] Cloud storage integration
- [ ] Mobile-responsive design

## License

MIT License - see main project LICENSE file

## Support

For issues or questions, please open an issue on GitHub.

---
**Built with ❤️ for UAE environmental monitoring**
