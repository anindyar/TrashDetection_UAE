#!/usr/bin/env python3
"""Test distance calculation logic for all_vehicles"""

import json

# Simulate the distance calculation
def get_box_center(bbox):
    """Calculate center of bounding box"""
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    return None

def calculate_pixel_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def pixels_to_feet(pixel_distance, altitude_ft, frame_width):
    """Convert pixel distance to real-world distance in feet"""
    # Simplified conversion (actual uses camera FOV)
    fov_rad = 1.466  # 84 degrees FOV
    ground_width_ft = 2 * altitude_ft * (fov_rad / 2)
    feet_per_pixel = ground_width_ft / frame_width
    return pixel_distance * feet_per_pixel

# Test with sample data
frame_vehicles = [
    {"vehicle_type": "car", "bbox": [0, 502, 379, 806], "confidence": 0.74, "plate_detected": False, "frame": 686}
]

waste_boxes = [
    [1689, 1017, 1729, 1087]  # Trash location
]

altitude_ft = 11.15
frame_width = 3840

print("Testing distance calculation logic:")
print("="*50)

for vehicle in frame_vehicles:
    vehicle_center = get_box_center(vehicle['bbox'])
    min_distance = float('inf')

    print(f"Vehicle: {vehicle['vehicle_type']}")
    print(f"  Vehicle center: {vehicle_center}")

    for waste_box in waste_boxes:
        waste_center = get_box_center(waste_box)
        print(f"  Trash center: {waste_center}")

        pixel_distance = calculate_pixel_distance(vehicle_center, waste_center)
        print(f"  Pixel distance: {pixel_distance:.2f} pixels")

        real_distance_ft = pixels_to_feet(pixel_distance, altitude_ft, frame_width)
        print(f"  Real distance: {real_distance_ft:.2f} feet")

        min_distance = min(min_distance, real_distance_ft)

    if min_distance != float('inf'):
        vehicle["distance_to_nearest_trash_ft"] = round(min_distance, 2)
    else:
        vehicle["distance_to_nearest_trash_ft"] = None

    print(f"  Final distance_to_nearest_trash_ft: {vehicle.get('distance_to_nearest_trash_ft')}")

print("\n" + "="*50)
print("Updated vehicle JSON:")
print(json.dumps(frame_vehicles[0], indent=2))
