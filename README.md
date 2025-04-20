# Impaired Driver Detection Model

🚗 **Part of the Heq.tech Platform – AI-Powered Road Safety**

This project is a core module of Heq.tech, a platform that uses real-time video analytics to detect impaired or dangerous drivers. Heq.tech analyzes CCTV traffic feeds and in-vehicle footage to identify erratic driving behavior and alert users to unsafe road conditions in their vicinity.

This repository focuses on the **Impaired Driver Detection Model**, which uses vehicle tracking, lane detection, and trajectory analysis to generate time-series data that feeds into a GRU-based classifier to detect potentially impaired driving behavior.

## Features

- Lane line detection using manual boundary marking  
- Vehicle detection and tracking  
- Homography transformation for bird's-eye view perspective  
- Linear position calculation relative to lane boundaries  
- FPS calculation and display  
- Video processing and saving capabilities  
- Generates trajectory data for ML analysis (used in impairment classification)

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can run the vehicle tracking and lane detection with the following command:

```bash
python process_video.py [input_video.mp4] --output-video output_video.mp4 --output-data trajectory_data.csv --homography --homography-path calibration.npy
```

Command-line arguments:
- `input_video.mp4` - Optional path to the input video file (defaults to test.mp4)
- `--output-video output_video.mp4` - Path to save the visualization video
- `--output-data trajectory_data.csv` - Path to save the trajectory data CSV
- `--show` - Show visualization during processing (default: True)
- `--homography` - Use homography transformation for bird's-eye view (default: True)
- `--homography-path calibration.npy` - Path to save/load homography calibration

You can press 'q' at any time to exit the video playback.

## Homography-based Road Flattening

The project now includes homography-based road flattening, which:
1. Allows selection of 4 points in the image to define the road plane
2. Transforms the perspective to create a bird's-eye view
3. Enables accurate linear position measurements
4. Shows both original view and transformed bird's-eye view

### How to Calibrate:
1. When running for the first time, you'll be prompted to select 4 points defining the road plane
   - Bottom-left corner of the road
   - Bottom-right corner of the road
   - Top-right corner of the road
   - Top-left corner of the road
2. After selecting the initial 4 points:
   - Press 'Enter' to continue to rectangle adjustment mode
   - Press 'a' to accept the points as they are without adjustment
3. In rectangle adjustment mode:
   - Drag any of the corner points to fine-tune their positions
   - Press 'r' to reset to your original selections
   - Press 'Enter' to confirm your adjustments
   - Press 'Esc' to cancel adjustments and use the original points
4. After calibration, the homography matrix can be saved for future use

You can also run just the homography testing tool to see the transformation:
```bash
python test_homography.py [input_video.mp4] --output output_video.mp4 --calibration calibration.npy
```

## Lane Definition

After homography calibration:
1. You'll be asked to enter the number of lanes
2. Then select points on the bird's-eye view image to define lane boundaries
3. For each boundary, select points from bottom to top and press Enter when done

## Trajectory Data

The output CSV file includes:
- Frame number and timestamp
- Vehicle ID, position, and class
- Lane number and relative position within the lane
- Warped coordinates in the bird's-eye view

## How It Works

The impaired driver detection algorithm follows these steps:

1. Apply homography transformation to obtain a bird's-eye view
2. Define lane boundaries in the transformed view
3. Detect vehicles in each frame
4. Track vehicles across frames
5. Transform vehicle positions to the bird's-eye view
6. Calculate linear position within lane
7. Record trajectory data for analysis

## Visualization

The output video shows:
- Green lines indicating the detected lane boundaries
- Colored boxes around detected vehicles
- Numerical indicators showing lane position
- A separate window with the bird's-eye view perspective 
