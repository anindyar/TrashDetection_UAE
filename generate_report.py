#!/usr/bin/env python3
"""
Generate human-readable reports from enhanced trash detection JSON output.
"""

import json
import argparse
from datetime import datetime
import csv

def generate_text_report(json_file, output_file):
    """Generate a text summary report."""

    with open(json_file, 'r') as f:
        data = json.load(f)

    total_frames = len(data)
    frames_with_waste = sum(1 for d in data if d['detections'])
    total_detections = sum(len(d['detections']) for d in data)

    # Count waste types
    waste_types = {}
    detections_with_vehicles = 0
    total_vehicles_detected = 0
    plates_captured = 0
    plate_numbers = []

    for frame_data in data:
        for detection in frame_data['detections']:
            waste_class = detection['class']
            waste_types[waste_class] = waste_types.get(waste_class, 0) + 1

            if detection['nearby_vehicles']:
                detections_with_vehicles += 1
                total_vehicles_detected += len(detection['nearby_vehicles'])

                for vehicle in detection['nearby_vehicles']:
                    if vehicle['plate_detected'] and 'plate_number' in vehicle:
                        plates_captured += 1
                        plate_numbers.append({
                            'frame': frame_data['frame'],
                            'plate': vehicle['plate_number'],
                            'vehicle': vehicle['vehicle_type'],
                            'distance': vehicle['distance_feet'],
                            'location': (detection['latitude'], detection['longitude'])
                        })

    # Generate report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("ENHANCED TRASH DETECTION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    report_lines.append("VIDEO SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Total frames processed: {total_frames}")
    report_lines.append(f"Frames with waste detected: {frames_with_waste}")
    report_lines.append(f"Total waste detections: {total_detections}")
    report_lines.append("")

    report_lines.append("WASTE BREAKDOWN")
    report_lines.append("-" * 80)
    for waste_type, count in sorted(waste_types.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {waste_type}: {count}")
    report_lines.append("")

    report_lines.append("VEHICLE PROXIMITY ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append(f"Waste detections with nearby vehicles: {detections_with_vehicles}")
    report_lines.append(f"Total vehicles detected near waste: {total_vehicles_detected}")
    report_lines.append(f"License plates captured: {plates_captured}")
    report_lines.append("")

    if plate_numbers:
        report_lines.append("LICENSE PLATES CAPTURED")
        report_lines.append("-" * 80)
        for i, plate_info in enumerate(plate_numbers, 1):
            report_lines.append(f"{i}. Frame {plate_info['frame']}:")
            report_lines.append(f"   Plate Number: {plate_info['plate']}")
            report_lines.append(f"   Vehicle Type: {plate_info['vehicle']}")
            report_lines.append(f"   Distance from waste: {plate_info['distance']:.2f} feet")
            report_lines.append(f"   GPS Location: {plate_info['location'][0]:.6f}, {plate_info['location'][1]:.6f}")
            report_lines.append("")

    report_lines.append("="*80)
    report_lines.append("DETAILED DETECTIONS")
    report_lines.append("="*80)

    for frame_data in data:
        if frame_data['detections']:
            report_lines.append(f"\nFrame {frame_data['frame']}:")
            for det in frame_data['detections']:
                report_lines.append(f"  - {det['class']} (confidence: {det['confidence']:.2f})")
                report_lines.append(f"    Location: {det['latitude']:.6f}, {det['longitude']:.6f}")

                if det['nearby_vehicles']:
                    report_lines.append(f"    Nearby vehicles: {len(det['nearby_vehicles'])}")
                    for vehicle in det['nearby_vehicles']:
                        vehicle_str = f"      * {vehicle['vehicle_type']} at {vehicle['distance_feet']:.2f}ft"
                        if vehicle['plate_detected'] and 'plate_number' in vehicle:
                            vehicle_str += f" - PLATE: {vehicle['plate_number']}"
                        report_lines.append(vehicle_str)
                else:
                    report_lines.append("    No vehicles nearby")

    report_lines.append("\n" + "="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"[INFO] Text report saved to: {output_file}")
    return report_lines

def generate_csv_report(json_file, output_file):
    """Generate a CSV report for easy data analysis."""

    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'waste_class', 'confidence', 'latitude', 'longitude',
                     'vehicles_nearby', 'vehicle_type', 'vehicle_distance_ft',
                     'plate_detected', 'plate_number', 'plate_confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for frame_data in data:
            for detection in frame_data['detections']:
                if detection['nearby_vehicles']:
                    for vehicle in detection['nearby_vehicles']:
                        row = {
                            'frame': frame_data['frame'],
                            'waste_class': detection['class'],
                            'confidence': detection['confidence'],
                            'latitude': detection['latitude'],
                            'longitude': detection['longitude'],
                            'vehicles_nearby': len(detection['nearby_vehicles']),
                            'vehicle_type': vehicle['vehicle_type'],
                            'vehicle_distance_ft': vehicle['distance_feet'],
                            'plate_detected': vehicle['plate_detected'],
                            'plate_number': vehicle.get('plate_number', ''),
                            'plate_confidence': vehicle.get('plate_confidence', '')
                        }
                        writer.writerow(row)
                else:
                    # Waste with no vehicles
                    row = {
                        'frame': frame_data['frame'],
                        'waste_class': detection['class'],
                        'confidence': detection['confidence'],
                        'latitude': detection['latitude'],
                        'longitude': detection['longitude'],
                        'vehicles_nearby': 0,
                        'vehicle_type': '',
                        'vehicle_distance_ft': '',
                        'plate_detected': False,
                        'plate_number': '',
                        'plate_confidence': ''
                    }
                    writer.writerow(row)

    print(f"[INFO] CSV report saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reports from enhanced trash detection JSON")
    parser.add_argument("--json", required=True, help="Path to detection JSON file")
    parser.add_argument("--output-txt", default="detection_report.txt", help="Output text report filename")
    parser.add_argument("--output-csv", default="detection_report.csv", help="Output CSV report filename")

    args = parser.parse_args()

    print("[INFO] Generating reports...")
    generate_text_report(args.json, args.output_txt)
    generate_csv_report(args.json, args.output_csv)
    print("[DONE] Reports generated successfully!")
