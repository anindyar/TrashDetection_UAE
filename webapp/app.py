#!/usr/bin/env python3
"""
Trash Detection Web Application
Flask backend for video upload, processing, and visualization
"""

import os
import json
import subprocess
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Create folders
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'MP4', 'srt', 'SRT'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle video and SRT file upload"""
    if 'video' not in request.files or 'srt' not in request.files:
        return jsonify({'error': 'Both video and SRT files required'}), 400

    video_file = request.files['video']
    srt_file = request.files['srt']

    if video_file.filename == '' or srt_file.filename == '':
        return jsonify({'error': 'No files selected'}), 400

    if not (allowed_file(video_file.filename) and allowed_file(srt_file.filename)):
        return jsonify({'error': 'Invalid file types'}), 400

    # Generate unique job ID
    job_id = str(uuid.uuid4())
    job_folder = Path(app.config['RESULTS_FOLDER']) / job_id
    job_folder.mkdir(exist_ok=True)

    # Save uploaded files
    video_path = job_folder / 'input_video.mp4'
    srt_path = job_folder / 'input_video.srt'

    video_file.save(video_path)
    srt_file.save(srt_path)

    return jsonify({
        'job_id': job_id,
        'message': 'Files uploaded successfully'
    })

@app.route('/process/<job_id>', methods=['POST'])
def process_video(job_id):
    """Run trash detection on uploaded video"""
    job_folder = Path(app.config['RESULTS_FOLDER']) / job_id

    if not job_folder.exists():
        return jsonify({'error': 'Job not found'}), 404

    video_path = job_folder / 'input_video.mp4'
    srt_path = job_folder / 'input_video.srt'
    output_json = job_folder / 'detections.json'
    output_video = job_folder / 'output_video.mp4'

    # Get parameters
    data = request.get_json() or {}
    conf_threshold = data.get('conf', 0.6)
    proximity_radius = data.get('radius', 15)

    # Run detection script
    script_path = Path(__file__).parent.parent / 'enhanced_trash_detect_v2.py'
    venv_python = Path(__file__).parent.parent / 'venv' / 'bin' / 'python'

    cmd = [
        str(venv_python),
        str(script_path),
        '--video', str(video_path),
        '--srt', str(srt_path),
        '--output_json', str(output_json),
        '--output_video', str(output_video),
        '--conf', str(conf_threshold),
        '--radius', str(proximity_radius)
    ]

    try:
        # Run in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Save process info
        with open(job_folder / 'process.pid', 'w') as f:
            f.write(str(process.pid))

        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'message': 'Detection started'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>')
def check_status(job_id):
    """Check processing status"""
    job_folder = Path(app.config['RESULTS_FOLDER']) / job_id
    output_json = job_folder / 'detections.json'

    if output_json.exists():
        return jsonify({'status': 'complete', 'job_id': job_id})

    pid_file = job_folder / 'process.pid'
    if pid_file.exists():
        return jsonify({'status': 'processing', 'job_id': job_id})

    return jsonify({'status': 'unknown', 'job_id': job_id})

@app.route('/results/<job_id>')
def get_results(job_id):
    """Get detection results with map data"""
    job_folder = Path(app.config['RESULTS_FOLDER']) / job_id
    output_json = job_folder / 'detections.json'

    if not output_json.exists():
        return jsonify({'error': 'Results not found'}), 404

    # Load detections
    with open(output_json, 'r') as f:
        detections = json.load(f)

    # Extract map markers (trash locations)
    markers = []
    for frame_data in detections:
        if frame_data.get('detections'):
            for detection in frame_data['detections']:
                marker = {
                    'frame': frame_data['frame'],
                    'lat': detection['latitude'],
                    'lon': detection['longitude'],
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'nearby_vehicles': detection.get('nearby_vehicles', [])
                }
                markers.append(marker)

    return jsonify({
        'job_id': job_id,
        'total_detections': len(markers),
        'markers': markers
    })

@app.route('/frame/<job_id>/<int:frame_num>')
def get_frame(job_id, frame_num):
    """Extract and return a specific frame as base64 image"""
    job_folder = Path(app.config['RESULTS_FOLDER']) / job_id
    video_path = job_folder / 'output_video.mp4'

    if not video_path.exists():
        return jsonify({'error': 'Video not found'}), 404

    # Extract frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': 'Frame not found'}), 404

    # Encode to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'frame': frame_num,
        'image': f'data:image/jpeg;base64,{frame_base64}'
    })

@app.route('/download/<job_id>/<filename>')
def download_file(job_id, filename):
    """Download processed files"""
    job_folder = Path(app.config['RESULTS_FOLDER']) / job_id
    return send_from_directory(job_folder, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
