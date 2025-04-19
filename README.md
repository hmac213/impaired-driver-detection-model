# Vehicle Trajectory Analysis

This project provides tools for analyzing vehicle trajectories from video footage, with a focus on detecting potential drunk driving behavior. It combines YOLOv8 for vehicle detection and tracking with computer vision techniques for lane detection.

## Features

- Vehicle detection and tracking using YOLOv8
- Lane detection using computer vision techniques
- Vehicle trajectory extraction relative to lane positions
- Visualization options for debugging and analysis
- Data export in CSV format for further analysis
- Support for multiple video formats including .mp4 and .mov files

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Process a video file and extract trajectory data:

```bash
# For MP4 files
python process_video.py path/to/video.mp4 --output-video output.mp4 --output-data trajectories.csv --show

# For MOV files
python process_video.py path/to/video.mov --output-video output.mov --output-data trajectories.csv --show
```

Arguments:
- `video_path`: Path to the input video file (supports .mp4, .mov, and other formats)
- `--output-video`: (Optional) Path to save visualization video (supports .mp4, .mov)
- `--output-data`: (Optional) Path to save trajectory data CSV
- `--show`: (Optional) Show visualization while processing

## Output Data Format

The trajectory data CSV contains the following columns:
- `frame`: Frame number in the video
- `time`: Time in seconds from the start of the video
- `vehicle_id`: Unique identifier for each tracked vehicle
- `x`, `y`: Vehicle center position in pixels
- `relative_lane_pos`: Position relative to lane boundaries (0 = left boundary, 1 = right boundary, 0.5 = center)
- `confidence`: Detection confidence score
- `vehicle_class`: Vehicle class ID (2 = car, 3 = motorcycle, 5 = bus, 7 = truck)

## Notes

- The lane detection algorithm works best with clear lane markings and good lighting conditions
- Vehicle tracking may occasionally lose track of vehicles during occlusions or rapid movements
- The relative lane position calculation assumes the camera is mounted with a good view of the road
- For best results, ensure the video footage has a clear view of both the vehicles and lane markings
- When using .mov files on macOS, the output video will use the H.264 codec for better compatibility

## Next Steps

1. Feed the extracted trajectory data into your neural network for drunk driving detection
2. Consider adding additional features such as:
   - Vehicle speed estimation
   - Multi-lane tracking
   - Turn signal detection
   - Vehicle orientation tracking 