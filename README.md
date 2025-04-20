# Lane Detection

This project implements lane detection in videos using traditional computer vision techniques.

## Features

- Lane line detection using Canny edge detection and Hough transform
- FPS calculation and display
- Video processing and saving capabilities

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can run the lane detection with the following command:

```bash
python lane_detector.py [input_video.mp4] --output output_video.mp4
```

The input video parameter is optional - if not provided, it will look for a file named "input_video.mp4" in the current directory.

You can press 'q' at any time to exit the video playback.

## Customization

You can customize the lane detection parameters by modifying the following in the `LaneDetector` class:

- `height_roi`: Controls the region of interest (lower portion of the image)
- `min_line_length`: Minimum line length for Hough transform
- `max_line_gap`: Maximum gap between line segments for Hough transform

## How It Works

The lane detection algorithm follows these steps:

1. Define region of interest (ROI) in the bottom portion of the frame
2. Convert ROI to grayscale and apply Gaussian blur
3. Apply Canny edge detection
4. Use a polygon mask to focus on the road area
5. Detect lines using Hough transform
6. Separate lines into left and right based on slope
7. Average and extrapolate the lane lines

## Visualization

The output video shows:
- Green lines indicating the detected lane boundaries
- FPS counter in the top-left corner 