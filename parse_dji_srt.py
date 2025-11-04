"""
Parse DJI SRT subtitle file to extract GPS, altitude, and telemetry data.
"""

import re
from typing import Dict, List, Optional

def parse_dji_srt(srt_file: str) -> List[Dict]:
    """
    Parse DJI SRT file and extract telemetry data per frame.

    Returns list of dicts with format:
    {
        'frame': int,
        'timestamp': str,
        'latitude': float,
        'longitude': float,
        'rel_alt': float,  # meters
        'abs_alt': float,  # meters
        'iso': int,
        'shutter': str,
        'fnum': float,
        'focal_len': float
    }
    """

    telemetry_data = []

    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by subtitle blocks (separated by double newlines)
    blocks = content.strip().split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')

        if len(lines) < 3:
            continue

        # Extract frame number from first line
        try:
            frame_num = int(lines[0].strip())
        except ValueError:
            continue

        # Find the line with telemetry data (contains brackets)
        telemetry_line = None
        timestamp = None

        for line in lines:
            if '[latitude:' in line or '[longitude:' in line:
                telemetry_line = line
            elif re.match(r'\d{4}-\d{2}-\d{2}', line):
                timestamp = line.strip()

        if not telemetry_line:
            continue

        # Extract values using regex
        frame_data = {
            'frame': frame_num,
            'timestamp': timestamp
        }

        # Parse all bracket values
        patterns = {
            'iso': r'\[iso:\s*(\d+)\]',
            'shutter': r'\[shutter:\s*([^\]]+)\]',
            'fnum': r'\[fnum:\s*([\d.]+)\]',
            'focal_len': r'\[focal_len:\s*([\d.]+)\]',
            'latitude': r'\[latitude:\s*([-\d.]+)\]',
            'longitude': r'\[longitude:\s*([-\d.]+)\]',
            'rel_alt': r'\[rel_alt:\s*([-\d.]+)',
            'abs_alt': r'\[abs_alt:\s*([-\d.]+)\]',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, telemetry_line)
            if match:
                value = match.group(1)
                # Convert to appropriate type
                if key in ['latitude', 'longitude', 'rel_alt', 'abs_alt', 'fnum', 'focal_len']:
                    frame_data[key] = float(value)
                elif key == 'iso':
                    frame_data[key] = int(value)
                else:
                    frame_data[key] = value

        telemetry_data.append(frame_data)

    return telemetry_data

def get_frame_telemetry(telemetry_data: List[Dict], frame_num: int) -> Optional[Dict]:
    """Get telemetry for a specific frame number."""
    for data in telemetry_data:
        if data['frame'] == frame_num:
            return data
    return None

def get_altitude_feet(telemetry_data: List[Dict], frame_num: int) -> float:
    """Get altitude in feet for a specific frame (default 100ft if not found)."""
    data = get_frame_telemetry(telemetry_data, frame_num)
    if data and 'rel_alt' in data:
        # Convert meters to feet
        return data['rel_alt'] * 3.28084
    return 100.0  # Default fallback

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_dji_srt.py <srt_file>")
        sys.exit(1)

    srt_file = sys.argv[1]
    telemetry = parse_dji_srt(srt_file)

    print(f"Parsed {len(telemetry)} frames from {srt_file}")
    print("\nFirst 5 frames:")
    for data in telemetry[:5]:
        print(f"Frame {data['frame']}: Lat={data.get('latitude')}, Lon={data.get('longitude')}, Alt={data.get('rel_alt')}m")

    print(f"\nLast frame: {telemetry[-1]['frame']}")
