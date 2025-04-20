import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import platform
import pandas as pd
from typing import List, Tuple
import time
from calibration import setup_calibration, VehicleDetector, LaneDetector
from video_preprocessing import preprocess_video

def get_video_writer(output_path, fps, frame_width, frame_height):
    """Create a video writer with the appropriate codec based on the platform and file extension."""
    output_path = str(output_path)
    ext = output_path.lower().split('.')[-1]
    
    if platform.system() == 'Darwin':  # macOS
        if ext == 'mov':
            # Try different codecs for macOS
            codecs = ['avc1', 'H264', 'mp4v', 'XVID']
            for codec in codecs:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                if out.isOpened():
                    return out
            # If no codec worked, try without specifying codec
            out = cv2.VideoWriter(output_path, -1, fps, (frame_width, frame_height))
        else:
            # For mp4, try mp4v first
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    else:  # Windows/Linux
        if ext == 'mov':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    return out

def process_video(video_path, output_path=None, show_visualization=False):
    """Process video and detect vehicles using YOLO."""
    # Perform setup steps
    vehicle_detector, lane_detector, video_info = setup_calibration(video_path)
    
    # Initialize tracking-related attributes
    tracks = {}  # Dictionary to store track information
    next_track_id = 1  # Counter for assigning track IDs
    track_start_times = {}  # Dictionary to store track start times
    all_track_data = []  # List to store track data for CSV output
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Initialize output video writer if needed
    out = None
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        out = get_video_writer(output_path, video_info['fps'], video_info['frame_width'], video_info['frame_height'])
        if not out or not out.isOpened():
            print(f"Warning: Could not create video writer for {output_path}. Visualization will not be saved.")
            out = None
    
    # Process frames
    pbar = tqdm(total=video_info['total_frames'], desc="Processing frames")
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect vehicles
        detections = vehicle_detector.detect(frame)
        
        # Simple tracking logic - associate detections with existing tracks
        # based on spatial proximity
        for det in detections:
            # Check if the vehicle's reference point is within the rectangle
            is_inside, relative_x = lane_detector.is_point_in_rectangle(det['center'])
            
            # Only process vehicles that are inside the rectangle
            if not is_inside:
                continue
                
            matched = False
            
            for track_id, track_history in tracks.items():
                if track_history:  # If track has previous detections
                    last_det = track_history[-1]
                    last_center = last_det['center']
                    
                    # Calculate distance between current detection and last point in track
                    dist = np.sqrt(
                        (det['center'][0] - last_center[0])**2 + 
                        (det['center'][1] - last_center[1])**2
                    )
                    
                    # If close enough, associate with this track
                    if dist < 50:  # Threshold for association
                        tracks[track_id].append(det)
                        
                        # Add track data with relative x-position
                        time_since_start = (frame_idx / video_info['fps']) - track_start_times[track_id]
                        
                        all_track_data.append({
                            'track_id': track_id,
                            'frame': frame_idx,
                            'time': time_since_start,
                            'x': relative_x,  # Now using relative_x as x
                            'y': det['center'][1],
                            'confidence': det['confidence'],
                            'class': det['class']
                        })
                        
                        matched = True
                        break
                        
            # If no match found, create new track
            if not matched:
                tracks[next_track_id] = [det]
                track_start_times[next_track_id] = frame_idx / video_info['fps']
                
                # Add initial track data
                all_track_data.append({
                    'track_id': next_track_id,
                    'frame': frame_idx,
                    'time': 0.0,
                    'x': relative_x,  # Now using relative_x as x
                    'y': det['center'][1],
                    'confidence': det['confidence'],
                    'class': det['class']
                })
                
                next_track_id += 1
        
        # Visualize if requested
        if show_visualization or (output_path and out is not None):
            vis_frame = frame.copy()
            
            # Draw homography rectangle
            if len(lane_detector.rectangle_points) == 4:
                rect_points = np.array(lane_detector.rectangle_points, np.int32)
                cv2.polylines(vis_frame, [rect_points], True, (255, 0, 0), 2)
            
            # Draw lane lines
            for coeffs in lane_detector.lanes:
                y_points = np.linspace(0, video_info['frame_height'], 100)
                x_points = np.polyval(coeffs, y_points)
                points = np.column_stack((x_points, y_points)).astype(np.int32)
                cv2.polylines(vis_frame, [points], False, (0, 0, 255), 2)
            
            # Draw current detections
            for det in detections:
                # Check if the vehicle's reference point is within the rectangle
                is_inside, relative_x = lane_detector.is_point_in_rectangle(det['center'])
                
                # Draw bounding box
                x1, y1, x2, y2 = det['bbox']
                color = (0, 255, 0) if is_inside else (0, 0, 255)  # Green if inside, red if outside
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw reference point
                ref_point = det['center']
                cv2.circle(vis_frame, ref_point, 5, color, -1)
                
                # Add text with class, confidence, and relative position
                if is_inside:
                    label = f"{det['class']}: {det['confidence']:.2f}, X: {relative_x:.2f}"
                else:
                    label = f"{det['class']}: {det['confidence']:.2f} (Outside ROI)"
                cv2.putText(vis_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if show_visualization:
                cv2.imshow('Vehicle Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            if out is not None:
                out.write(vis_frame)
                
        frame_idx += 1
        pbar.update(1)
        
    # Clean up
    pbar.close()
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Save track data to CSV
    if all_track_data:
        df = pd.DataFrame(all_track_data)
        csv_path = Path(output_path).with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"Track data saved to {csv_path}")
    
    return tracks

def main():
    parser = argparse.ArgumentParser(description='Detect vehicles in video using YOLOv8')
    parser.add_argument('video_path', help='Path to input video file', nargs='?', default='test.mp4')
    parser.add_argument('--output-video', help='Path to output visualization video')
    parser.add_argument('--show', action='store_true', help='Show visualization while processing', default=True)
    parser.add_argument('--model', help='Path to YOLOv8 model', default='yolov8n.pt')
    parser.add_argument('--conf', type=float, help='Confidence threshold', default=0.5)
    args = parser.parse_args()
    
    # Convert paths to Path objects for better handling
    video_path = Path(args.video_path)
    output_video = Path(args.output_video) if args.output_video else None
    
    # Verify input file exists
    if not video_path.exists():
        print(f"Input video file not found: {video_path}")
        video_path = Path('test.mp4')  # Try using test.mp4 as fallback
        if not video_path.exists():
            raise FileNotFoundError(f"Default video file not found: {video_path}")
        print(f"Using default video file: {video_path}")
    
    # Process video
    tracks = process_video(video_path,
                         output_path=output_video,
                         show_visualization=args.show)
    
    print(f"Detected {len(tracks)} vehicle tracks")
        
if __name__ == '__main__':
    main() 