import cv2
import numpy as np
import pandas as pd
from vehicle_detector import VehicleDetector
from pathlib import Path
import argparse
from tqdm import tqdm
import platform
from scipy.interpolate import interp1d

def compute_screen_relative_position(vehicle_center, frame_width):
    """
    Compute vehicle's position relative to screen center.
    Returns a value between -1 and 1, where:
    - 0 means perfectly centered in frame
    - -1 means on far right of frame
    - 1 means on far left of frame
    """
    x, _ = vehicle_center
    screen_center = frame_width / 2
    
    # Calculate position relative to screen center
    # Normalize to [-1, 1] where negative is right of center, positive is left
    relative_pos = (x - screen_center) / (screen_center)
    
    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, relative_pos))

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

class LaneSelector:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.points = []
        self.current_line = []
        self.lines = []
        self.window_name = "Lane Selection"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_line.append((x, y))
            cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(self.window_name, self.frame)
            
    def get_lane_points(self, num_lanes, has_barrier=False):
        """Get user-selected points for lane boundaries."""
        num_lines = num_lanes + 1
        if has_barrier:
            num_lines += 1
            
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        original_frame = self.frame.copy()
        
        for i in range(num_lines):
            self.current_line = []
            self.frame = original_frame.copy()
            
            # Draw existing lines
            for line_points in self.lines:
                for j in range(len(line_points) - 1):
                    cv2.line(self.frame, line_points[j], line_points[j+1], (0, 255, 0), 2)
            
            print(f"\nClick points for lane line {i+1} from bottom to top. Press 'Enter' when done with this line.")
            cv2.imshow(self.window_name, self.frame)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter key
                    if len(self.current_line) >= 3:  # Minimum points for quadratic fit
                        self.lines.append(self.current_line)
                        break
                    else:
                        print("Please select at least 3 points for each line")
                elif key == 27:  # ESC key
                    cv2.destroyWindow(self.window_name)
                    return None
                    
        cv2.destroyWindow(self.window_name)
        return self.lines

def fit_lane_polynomials(lane_points, frame_height):
    """Fit polynomials to lane boundary points."""
    polynomials = []
    for line_points in lane_points:
        x = np.array([p[0] for p in line_points])
        y = np.array([p[1] for p in line_points])
        
        # Fit quadratic polynomial
        coeffs = np.polyfit(y, x, 2)
        polynomials.append(coeffs)
        
    return polynomials

def get_lane_boundaries(polynomials, y):
    """Get x-coordinates of lane boundaries at given y-coordinate."""
    x_coords = []
    for coeffs in polynomials:
        x = coeffs[0] * y**2 + coeffs[1] * y + coeffs[2]
        x_coords.append(int(x))
    return x_coords

def compute_lane_relative_position(vehicle_center, lane_polynomials, num_lanes):
    """
    Compute vehicle's position relative to lane boundaries.
    Returns (lane_number, relative_position) where:
    - lane_number is 0-based index of the lane
    - relative_position is between -1 and 1 (-1 = left edge, 1 = right edge)
    """
    x, y = vehicle_center
    
    # Get x-coordinates of all lane boundaries at vehicle's y-position
    lane_x_coords = get_lane_boundaries(lane_polynomials, y)
    
    # Find which lane the vehicle is in
    for i in range(len(lane_x_coords) - 1):
        left_x = lane_x_coords[i]
        right_x = lane_x_coords[i + 1]
        
        if left_x <= x <= right_x or right_x <= x <= left_x:
            # Found the lane
            lane_width = abs(right_x - left_x)
            lane_center = (left_x + right_x) / 2
            
            # Calculate relative position in lane (-1 to 1)
            relative_pos = 2 * (x - lane_center) / lane_width
            return i, max(-1.0, min(1.0, relative_pos))
            
    # If vehicle is outside all lanes, find nearest lane
    distances = [abs(x - lane_x) for lane_x in lane_x_coords]
    nearest_boundary = np.argmin(distances)
    
    if nearest_boundary == 0:
        return 0, -1.0
    elif nearest_boundary == len(lane_x_coords) - 1:
        return num_lanes - 1, 1.0
    else:
        return nearest_boundary - 1, 1.0 if x > lane_x_coords[nearest_boundary] else -1.0

def draw_lane_boundaries(frame, lane_polynomials):
    """Draw lane boundary lines on the frame."""
    height = frame.shape[0]
    y_coords = np.linspace(0, height - 1, 100)
    
    for coeffs in lane_polynomials:
        points = []
        for y in y_coords:
            x = coeffs[0] * y**2 + coeffs[1] * y + coeffs[2]
            points.append([int(x), int(y)])
            
        points = np.array(points)
        cv2.polylines(frame, [points], False, (0, 255, 0), 2)
        
    return frame

def process_video(video_path, output_path=None, show_visualization=False):
    """Process video and extract vehicle trajectory data."""
    # Initialize detector
    vehicle_detector = VehicleDetector()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read first frame for lane selection
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
        
    # Get lane configuration from user
    num_lanes = int(input("Enter the number of lanes: "))
    has_barrier = input("Is there a barrier in the middle? (y/n): ").lower() == 'y'
    
    # Get lane boundary points
    lane_selector = LaneSelector(first_frame)
    lane_points = lane_selector.get_lane_points(num_lanes, has_barrier)
    if lane_points is None:
        print("Lane selection cancelled")
        return None
        
    # Fit polynomials to lane boundaries
    lane_polynomials = fit_lane_polynomials(lane_points, frame_height)
    
    # Initialize output video writer if needed
    out = None
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        out = get_video_writer(output_path, fps, frame_width, frame_height)
        if not out or not out.isOpened():
            print(f"Warning: Could not create video writer for {output_path}. Visualization will not be saved.")
            print("Try changing the output format to .mp4 instead of .mov")
            out = None
                            
    # Initialize data storage
    trajectory_data = []
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing frames")
    frame_idx = 0
    
    # Reset video capture to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect vehicles
        detections, tracks = vehicle_detector.detect_and_track(frame)
        
        # Process each detection
        for det in detections:
            lane_num, relative_pos = compute_lane_relative_position(det['center'], lane_polynomials, num_lanes)
            
            # Store trajectory data
            trajectory_data.append({
                'frame': frame_idx,
                'time': frame_idx / fps,
                'vehicle_id': None,  # Will be filled in post-processing
                'x': det['center'][0],
                'y': det['center'][1],
                'lane_number': lane_num,
                'relative_pos': relative_pos,  # Between -1 and 1 within lane
                'confidence': det['confidence'],
                'vehicle_class': det['class']
            })
            
        # Visualize if requested
        if show_visualization or (output_path and out is not None):
            vis_frame = frame.copy()
            
            # Draw lane boundaries
            vis_frame = draw_lane_boundaries(vis_frame, lane_polynomials)
            
            # Draw detections and positions
            for det in detections:
                # Draw bounding box
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw position indicator
                x, y = det['center']
                lane_num, rel_pos = compute_lane_relative_position(det['center'], lane_polynomials, num_lanes)
                
                # Color based on distance from center
                color = (0, 255, 0) if abs(rel_pos) < 0.5 else (0, 165, 255)
                
                # Draw position marker and value
                cv2.circle(vis_frame, (int(x), int(y)), 5, color, -1)
                cv2.putText(vis_frame, f"L{lane_num}:{rel_pos:.2f}", (int(x) + 10, int(y) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if show_visualization:
                cv2.imshow('Processing', vis_frame)
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
    
    # Convert trajectory data to DataFrame
    df = pd.DataFrame(trajectory_data)
    
    # Post-process to assign consistent vehicle IDs based on tracking
    for track_id, track_data in vehicle_detector.tracks.items():
        for point in track_data:
            mask = (df['x'] == point['center'][0]) & (df['y'] == point['center'][1])
            df.loc[mask, 'vehicle_id'] = track_id
            
    return df

def main():
    parser = argparse.ArgumentParser(description='Process video for vehicle trajectory analysis')
    parser.add_argument('video_path', help='Path to input video file (supports .mp4, .mov, and other formats)')
    parser.add_argument('--output-video', help='Path to output visualization video (supports .mp4, .mov)')
    parser.add_argument('--output-data', help='Path to save trajectory data CSV')
    parser.add_argument('--show', action='store_true', help='Show visualization while processing')
    args = parser.parse_args()
    
    # Convert paths to Path objects for better handling
    video_path = Path(args.video_path)
    output_video = Path(args.output_video) if args.output_video else None
    output_data = Path(args.output_data) if args.output_data else None
    
    # Verify input file exists
    if not video_path.exists():
        raise FileNotFoundError(f"Input video file not found: {video_path}")
    
    # Process video
    df = process_video(video_path,
                      output_path=output_video,
                      show_visualization=args.show)
                      
    # Save trajectory data
    if output_data:
        df.to_csv(output_data, index=False)
        print(f"Trajectory data saved to: {output_data}")
        
if __name__ == '__main__':
    main() 