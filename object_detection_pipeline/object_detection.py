import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import platform
import pandas as pd
from typing import List, Tuple, Dict
import time
from object_detection_pipeline.calibration import setup_processing, VehicleDetector, LaneDetector
from object_detection_pipeline.video_preprocessing import preprocess_video

# Create output directory if it doesn't exist
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

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

class VehicleTracker:
    def __init__(self, max_frames_missing: int = 3, max_position_diff: float = 0.3, max_velocity_diff: float = 0.2):
        self.tracks = {}  # Dictionary to store track information
        self.next_track_id = 1
        self.track_start_times = {}
        self.max_frames_missing = max_frames_missing
        self.max_position_diff = max_position_diff
        self.max_velocity_diff = max_velocity_diff
        
    def _calculate_velocity(self, track_history: List[dict]) -> Tuple[float, float]:
        """Calculate velocity from the last few points in the track."""
        if len(track_history) < 2:
            return 0.0, 0.0
            
        # Use the last 3 points if available, otherwise use all points
        points = track_history[-3:] if len(track_history) >= 3 else track_history
        
        # Calculate average velocity
        dx = np.mean([p['x'] - points[i-1]['x'] for i, p in enumerate(points[1:], 1)])
        dy = np.mean([p['y'] - points[i-1]['y'] for i, p in enumerate(points[1:], 1)])
        
        return dx, dy
        
    def _extrapolate_position(self, track_history: List[dict], frames_ahead: int) -> Tuple[float, float]:
        """Extrapolate position based on velocity."""
        if len(track_history) < 2:
            return track_history[-1]['x'], track_history[-1]['y']
            
        dx, dy = self._calculate_velocity(track_history)
        last_point = track_history[-1]
        
        return last_point['x'] + dx * frames_ahead, last_point['y'] + dy * frames_ahead
        
    def _find_matching_track(self, detection: dict, frame_idx: int) -> int:
        """Find a matching track for the detection, considering extrapolation."""
        best_match_id = None
        min_distance = float('inf')
        
        for track_id, track_info in self.tracks.items():
            # Skip if track is too old
            if frame_idx - track_info['last_frame'] > self.max_frames_missing:
                continue
                
            # Skip if vehicle class doesn't match
            if track_info['history'][-1]['class'] != detection['class']:
                continue
                
            # Calculate frames since last seen
            frames_missing = frame_idx - track_info['last_frame']
            
            # Extrapolate position
            pred_x, pred_y = self._extrapolate_position(track_info['history'], frames_missing)
            
            # Calculate position difference
            pos_diff = np.sqrt((detection['x'] - pred_x)**2 + (detection['y'] - pred_y)**2)
            
            # Calculate velocity difference
            dx, dy = self._calculate_velocity(track_info['history'])
            new_dx = detection['x'] - track_info['history'][-1]['x']
            new_dy = detection['y'] - track_info['history'][-1]['y']
            vel_diff = np.sqrt((dx - new_dx)**2 + (dy - new_dy)**2)
            
            # Combined score with more weight on position difference
            score = pos_diff * 2 + vel_diff
            
            # Check if this is a better match
            if score < min_distance and pos_diff < self.max_position_diff and vel_diff < self.max_velocity_diff:
                min_distance = score
                best_match_id = track_id
                
        return best_match_id
        
    def update(self, detections: List[dict], frame_idx: int) -> List[dict]:
        """Update tracks with new detections."""
        updated_detections = []
        
        # First, update existing tracks
        for track_id, track_info in list(self.tracks.items()):
            # If track is too old, remove it
            if frame_idx - track_info['last_frame'] > self.max_frames_missing:
                del self.tracks[track_id]
                continue
                
        # Sort detections by confidence to process most confident ones first
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Process new detections
        for det in detections:
            # Try to find a matching track
            matching_track_id = self._find_matching_track(det, frame_idx)
            
            if matching_track_id is not None:
                # Update existing track
                det['track_id'] = matching_track_id
                self.tracks[matching_track_id]['history'].append(det)
                self.tracks[matching_track_id]['last_frame'] = frame_idx
            else:
                # Only create new track if we don't have too many already
                if len(self.tracks) < 10:  # Limit total number of tracks
                    det['track_id'] = self.next_track_id
                    self.tracks[self.next_track_id] = {
                        'history': [det],
                        'last_frame': frame_idx
                    }
                    self.track_start_times[self.next_track_id] = frame_idx
                    self.next_track_id += 1
                else:
                    # Skip this detection if we have too many tracks
                    continue
                
            updated_detections.append(det)
            
        return updated_detections

def process_video(video_path, show_visualization=False):
    """Process video and detect vehicles using YOLO."""
    # Perform setup steps
    vehicle_detector, lane_detector, video_info = setup_processing(video_path)
    
    # Initialize tracker
    tracker = VehicleTracker()
    all_track_data = []  # List to store track data for CSV output
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Initialize output video writer
    video_name = Path(video_path).stem
    output_video_path = OUTPUT_DIR / f"{video_name}_processed.mp4"
    out = get_video_writer(str(output_video_path), video_info['fps'], video_info['frame_width'], video_info['frame_height'])
    if not out or not out.isOpened():
        print(f"Warning: Could not create video writer for {output_video_path}. Visualization will not be saved.")
        out = None
    
    # Set target processing frame rate (e.g., 10 fps)
    target_fps = 10
    frame_skip = int(video_info['fps'] / target_fps)
    if frame_skip < 1:
        frame_skip = 1
    
    print(f"Original FPS: {video_info['fps']}, Processing at {target_fps} FPS (skipping {frame_skip-1} frames)")
    
    # Process frames
    pbar = tqdm(total=video_info['total_frames'] // frame_skip, desc="Processing frames")
    frame_idx = 0
    processed_frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames to achieve target FPS
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
            
        # Detect vehicles
        detections = vehicle_detector.detect(frame)
        
        # Process detections
        processed_detections = []
        for det in detections:
            # Check if the vehicle's reference point is within the rectangle
            is_inside, relative_x = lane_detector.is_point_in_rectangle(det['center'])
            
            # Only process vehicles that are inside the rectangle
            if not is_inside:
                continue
                
            # Get the normalized position within the lane
            relative_y, _ = lane_detector.get_lane_position(det['center'])
            
            # Add position information to detection
            det['x'] = relative_x
            det['y'] = relative_y
            processed_detections.append(det)
            
        # Update tracks
        updated_detections = tracker.update(processed_detections, processed_frame_idx)
        
        # Add track data
        for det in updated_detections:
            time_since_start = (processed_frame_idx / target_fps) - tracker.track_start_times[det['track_id']]
            all_track_data.append({
                'track_id': det['track_id'],
                'frame': processed_frame_idx,
                'time': time_since_start,
                'x': det['x'],
                'y': det['y'],
                'confidence': det['confidence'],
                'class': det['class']
            })
            
        # Visualize if requested
        if show_visualization or out is not None:
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
            for det in updated_detections:
                # Draw bounding box
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw reference point
                ref_point = det['center']
                cv2.circle(vis_frame, ref_point, 5, (0, 255, 0), -1)
                
                # Add text with class, confidence, and relative position
                label = f"{det['class']}: {det['confidence']:.2f}, X: {det['x']:.2f}, Y: {det['y']:.2f}"
                cv2.putText(vis_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if show_visualization:
                cv2.imshow('Vehicle Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            if out is not None:
                out.write(vis_frame)
                
        frame_idx += 1
        processed_frame_idx += 1
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
        csv_path = OUTPUT_DIR / f"{video_name}_tracks.csv"
        df.to_csv(csv_path, index=False)
        print(f"Track data saved to {csv_path}")
    
    return tracker.tracks

def main():
    parser = argparse.ArgumentParser(description='Detect vehicles in video using YOLOv8')
    parser.add_argument('video_path', help='Path to input video file', nargs='?', default='test.mp4')
    parser.add_argument('--show', action='store_true', help='Show visualization while processing', default=True)
    parser.add_argument('--model', help='Path to YOLOv8 model', default='yolov8n.pt')
    parser.add_argument('--conf', type=float, help='Confidence threshold', default=0.5)
    args = parser.parse_args()
    
    # Convert paths to Path objects for better handling
    video_path = Path(args.video_path)
    
    # Verify input file exists
    if not video_path.exists():
        print(f"Input video file not found: {video_path}")
        video_path = Path('test.mp4')  # Try using test.mp4 as fallback
        if not video_path.exists():
            raise FileNotFoundError(f"Default video file not found: {video_path}")
        print(f"Using default video file: {video_path}")
    
    # Process video
    tracks = process_video(video_path, show_visualization=args.show)
    
    print(f"Detected {len(tracks)} vehicle tracks")
        
if __name__ == '__main__':
    main() 