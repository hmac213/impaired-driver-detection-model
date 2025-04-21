import requests
import xml.etree.ElementTree as ET
import cv2
import sys
import os
import uuid
import json
import numpy as np
from pymongo import MongoClient
import datetime
import time
from object_detection_pipeline.calibration import LaneDetector, VehicleDetector, setup_processing

# Add warning filter to suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Create a custom JSON encoder to handle NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

# MongoDB connection
MONGO_URI = "mongodb+srv://Temp:hack@cluster0.lhazyyp.mongodb.net/DriverData?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["DriverData"]
calibration_collection = db["camera_calibration"]
camera_collection = db["camera_db"]

# Add VehicleTracker from object_detection.py
class VehicleTracker:
    def __init__(self, max_frames_missing: int = 3, max_position_diff: float = 0.3, max_velocity_diff: float = 0.2):
        self.tracks = {}  # Dictionary to store track information
        self.next_track_id = 1
        self.track_start_times = {}
        self.max_frames_missing = max_frames_missing
        self.max_position_diff = max_position_diff
        self.max_velocity_diff = max_velocity_diff
        
    def _calculate_velocity(self, track_history: list) -> tuple:
        """Calculate velocity from the last few points in the track."""
        if len(track_history) < 2:
            return 0.0, 0.0
            
        # Use the last 3 points if available, otherwise use all points
        points = track_history[-3:] if len(track_history) >= 3 else track_history
        
        # Calculate average velocity
        dx = np.mean([p['x'] - points[i-1]['x'] for i, p in enumerate(points[1:], 1)])
        dy = np.mean([p['y'] - points[i-1]['y'] for i, p in enumerate(points[1:], 1)])
        
        return dx, dy
        
    def _extrapolate_position(self, track_history: list, frames_ahead: int) -> tuple:
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
        
        # Add x and y to detection for compatibility with tracker
        detection['x'] = detection['center'][0]
        detection['y'] = detection['center'][1]
        
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
        
    def update(self, detections: list, frame_idx: int) -> list:
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
            # Add x and y to detection for compatibility with tracker
            det['x'] = det['center'][0]
            det['y'] = det['center'][1]
            
            # Try to find a matching track
            matching_track_id = self._find_matching_track(det, frame_idx)
            
            if matching_track_id is not None:
                # Update existing track
                det['track_id'] = matching_track_id
                self.tracks[matching_track_id]['history'].append(det)
                self.tracks[matching_track_id]['last_frame'] = frame_idx
            else:
                # Create new track
                det['track_id'] = self.next_track_id
                self.tracks[self.next_track_id] = {
                    'history': [det],
                    'last_frame': frame_idx
                }
                self.track_start_times[self.next_track_id] = frame_idx
                self.next_track_id += 1
                
            updated_detections.append(det)
            
        return updated_detections

def get_camera_stream(camera_id="I80_Northgate"):
    """Fetch streaming URL for specified camera ID"""
    # 1. Download the District 3 CCTV status XML
    status_url = "https://cwwp2.dot.ca.gov/data/d3/cctv/cctvStatusD03.xml"
    resp = requests.get(status_url)
    resp.raise_for_status()

    # 2. Parse it
    root = ET.fromstring(resp.content)

    # 3. Find the camera based on ID (default is I-80 Northgate)
    if camera_id == "I80_Northgate":
        search_route = "I-80"
        search_location = "Northgate"
    else:
        # For future support of other cameras
        search_route = camera_id.split('_')[0].replace('I', 'I-')
        search_location = camera_id.split('_')[1]

    for cam in root.findall(".//cctv"):
        # Safely extract route and locationName from the <location> element
        route_elem = cam.find("location/route")
        route = route_elem.text if route_elem is not None else ""
        loc_elem = cam.find("location/locationName")
        loc = loc_elem.text if loc_elem is not None else ""
        
        if search_route in route and search_location in loc:
            url_elem = cam.find("imageData/streamingVideoURL")
            if url_elem is not None and url_elem.text.startswith("http"):
                return url_elem.text
    
    raise RuntimeError(f"Could not find camera {camera_id} in XML")

def get_or_create_calibration(camera_id):
    """Get existing calibration from database or create new one"""
    # Check if calibration exists
    calibration = calibration_collection.find_one(
        {"camera_id": camera_id, "is_active": True},
        {"_id": 0}
    )
    
    if calibration:
        print(f"Found existing calibration for camera {camera_id}")
        return calibration
    
    print(f"No calibration found for camera {camera_id}, will create new calibration")
    return None

def store_calibration(camera_id, vehicle_detector, lane_detector):
    """Store calibration data in MongoDB"""
    calibration_id = str(uuid.uuid4())
    # Fix the deprecation warning by using datetime.now(datetime.UTC)
    try:
        # Python 3.11+ has datetime.UTC
        timestamp = datetime.datetime.now(datetime.UTC)
    except AttributeError:
        # Fallback for older Python versions
        timestamp = datetime.datetime.now(datetime.timezone.utc)
    
    # Extract data from detectors
    lane_polynomials = lane_detector.lanes
    lane_points = lane_detector.points
    rectangle_points = lane_detector.rectangle_points
    entrance_side = lane_detector.entrance_side if hasattr(lane_detector, 'entrance_side') else 0
    exit_side = lane_detector.exit_side if hasattr(lane_detector, 'exit_side') else 2
    
    # Convert numpy arrays to lists for MongoDB storage
    if isinstance(lane_polynomials, np.ndarray):
        lane_polynomials = lane_polynomials.tolist()
    if isinstance(rectangle_points, np.ndarray):
        rectangle_points = rectangle_points.tolist()
    
    # Ensure all nested arrays are also converted to lists
    if isinstance(lane_polynomials, list):
        for i, poly in enumerate(lane_polynomials):
            if isinstance(poly, np.ndarray):
                lane_polynomials[i] = poly.tolist()
    
    # Convert matrices to lists
    homography_matrix = lane_detector.homography_matrix.tolist() if lane_detector.homography_matrix is not None else []
    inv_homography_matrix = lane_detector.inv_homography_matrix.tolist() if hasattr(lane_detector, 'inv_homography_matrix') else []
    
    # Reference point
    ref_point = vehicle_detector.ref_point if vehicle_detector.ref_point else (0.5, 1.0)
    
    # Create calibration document
    calibration_data = {
        "calibration_id": calibration_id,
        "camera_id": camera_id,
        "location_name": f"Location for camera {camera_id}",
        "timestamp": timestamp,
        "created_at": timestamp,
        
        # Lane data
        "lane_data": {
            "lane_polynomials": lane_polynomials,
            "lane_points": lane_points if isinstance(lane_points, list) else []
        },
        
        # ROI data
        "roi_data": {
            "rectangle_points": rectangle_points,
            "entrance_side": entrance_side,
            "exit_side": exit_side,
            "world_width": lane_detector.world_width,
            "world_length": lane_detector.world_length
        },
        
        # Homography data
        "homography_data": {
            "matrix": homography_matrix,
            "inverse_matrix": inv_homography_matrix
        },
        
        # Reference point for vehicle detection
        "reference_point": {
            "rel_x": float(ref_point[0]),
            "rel_y": float(ref_point[1])
        },
        
        # Set this calibration as active
        "is_active": True
    }
    
    # Deactivate previous calibrations
    calibration_collection.update_many(
        {"camera_id": camera_id, "is_active": True},
        {"$set": {"is_active": False}}
    )
    
    # Insert new calibration
    result = calibration_collection.insert_one(calibration_data)
    
    if result.acknowledged:
        print(f"Stored new calibration with ID: {calibration_id}")
        return calibration_data
    else:
        print("Failed to store calibration")
        return None

def apply_calibration_to_detectors(calibration, vehicle_detector, lane_detector):
    """Apply stored calibration data to detectors"""
    # Set reference point for vehicle detector
    ref_point = calibration["reference_point"]
    vehicle_detector.ref_point = (ref_point["rel_x"], ref_point["rel_y"])
    
    # Set lane data
    lane_data = calibration["lane_data"]
    lane_detector.lanes = lane_data["lane_polynomials"]
    lane_detector.points = lane_data["lane_points"]
    
    # Set ROI data
    roi_data = calibration["roi_data"]
    lane_detector.rectangle_points = roi_data["rectangle_points"]
    lane_detector.entrance_side = roi_data["entrance_side"]
    lane_detector.exit_side = roi_data["exit_side"]
    lane_detector.world_width = roi_data["world_width"]
    lane_detector.world_length = roi_data["world_length"]
    
    # Set homography matrix
    homography_data = calibration["homography_data"]
    lane_detector.homography_matrix = np.array(homography_data["matrix"])
    lane_detector.inv_homography_matrix = np.array(homography_data["inverse_matrix"])
    
    print("Applied stored calibration to detectors")
    return vehicle_detector, lane_detector

def main():
    # Camera ID
    camera_id = "I80_Northgate"
    
    try:
        # Get streaming URL
        print(f"Getting stream URL for camera {camera_id}...")
        stream_url = get_camera_stream(camera_id)
        print(f"Streaming URL: {stream_url}")
        
        # Initialize object detector and tracker
        vehicle_detector = VehicleDetector()
        lane_detector = LaneDetector()
        tracker = VehicleTracker()  # Add vehicle tracker
        
        # Check for existing calibration
        calibration = get_or_create_calibration(camera_id)
        
        # Open the video stream
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video stream")
            
        # Set buffer size to minimize buffering
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Capture a frame for initial setup if needed
        ret, init_frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read initial frame from stream")
        
        if calibration:
            # Apply stored calibration
            vehicle_detector, lane_detector = apply_calibration_to_detectors(
                calibration, vehicle_detector, lane_detector
            )
        else:
            # Perform setup using the first frame
            print("No existing calibration found. Please set up lane detection now.")
            print("Press 'q' after completing the setup")
            
            # Get reference point
            vehicle_detector.get_reference_point(init_frame, None)
            
            # Get homography rectangle
            lane_detector.get_homography_rectangle(init_frame)
            
            # Get lane points
            lane_detector.get_lane_points(init_frame)
            
            # Store calibration in database
            calibration = store_calibration(camera_id, vehicle_detector, lane_detector)
        
        print("Starting object detection on live stream...")
        frame_count = 0
        
        # Target fps for processing (adjust this value as needed)
        target_fps = 10
        frame_interval = 1.0 / target_fps
        
        while True:
            # Record start time for this iteration
            start_time = time.time()
            
            # Clear buffer by reading frames until we get to the newest one
            for _ in range(5):  # Limit to 5 attempts to avoid infinite loop
                cap.grab()  # Just grab frame without decoding
                
            # Now read the latest frame
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or error")
                break
            
            # Process frame
            # Detect vehicles
            detections = vehicle_detector.detect(frame)
            
            # Update tracking
            tracked_detections = tracker.update(detections, frame_count)
            
            # Draw ROI rectangle
            if lane_detector.rectangle_points and len(lane_detector.rectangle_points) == 4:
                points = np.array(lane_detector.rectangle_points, dtype=np.int32)
                cv2.polylines(frame, [points], True, (0, 255, 0), 2)
            
            # Draw lane lines
            if lane_detector.lanes:
                height, width = frame.shape[:2]
                for coeffs in lane_detector.lanes:
                    # Draw polynomial curve
                    y_points = np.linspace(0, height-1, 50)
                    x_points = np.polyval(coeffs, y_points)
                    
                    # Filter points within image boundaries
                    valid_indices = (x_points >= 0) & (x_points < width)
                    y_valid = y_points[valid_indices].astype(np.int32)
                    x_valid = x_points[valid_indices].astype(np.int32)
                    
                    # Draw lane line
                    for i in range(len(x_valid) - 1):
                        cv2.line(frame, (x_valid[i], y_valid[i]), 
                                (x_valid[i+1], y_valid[i+1]), (0, 0, 255), 2)
            
            # Process tracked detections and draw bounding boxes
            for det in tracked_detections:
                x1, y1, x2, y2 = det['bbox']
                center = det['center']
                class_name = det['class']
                confidence = det['confidence']
                track_id = det.get('track_id', '')
                
                # Get lane position
                in_roi, relative_pos = lane_detector.is_point_in_rectangle(center)
                
                # Draw bounding box and information
                color = (255, 0, 0)  # Default color: blue
                if track_id in tracker.tracks and len(tracker.tracks[track_id]['history']) > 5:
                    # Use green for stable tracks
                    color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw reference point
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Add text with detection info
                label = f"{class_name} {track_id} {confidence:.2f}"
                if in_roi:
                    position_text = f"Pos: {relative_pos:.2f}"
                    cv2.putText(frame, position_text, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Store detection in database
                    detection_data = {
                        "camera_id": camera_id,
                        "track_id": str(track_id),
                        "class": class_name,
                        "frame_number": frame_count,
                        "x": float(center[0]),
                        "y": float(center[1]),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        # Fix the deprecation warning by using datetime.now(datetime.UTC)
                        "timestamp": datetime.datetime.now(datetime.timezone.utc),
                        "calibration_id": calibration["calibration_id"],
                        "position_in_lane": float(relative_pos)
                    }
                    
                    # Uncomment to store each detection (may generate a lot of data)
                    # camera_collection.insert_one(detection_data)
                
                cv2.putText(frame, label, (x1, y1-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Display frame with detections
            cv2.imshow("Live Highway Feed with Object Detection", frame)
            
            # Increment frame counter
            frame_count += 1
            
            # Calculate elapsed time and sleep to maintain target FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed_time)
            
            # Add FPS information to the frame
            actual_fps = 1.0 / (elapsed_time + sleep_time if sleep_time > 0 else elapsed_time)
            cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame again with FPS info
            cv2.imshow("Live Highway Feed with Object Detection", frame)
            
            # Wait for the calculated time to maintain frame rate
            # Use a small value for waitKey during the sleep time for responsiveness
            remaining_ms = int(sleep_time * 1000)
            if remaining_ms <= 0:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Playback interrupted by user")
                    break
            else:
                # Break the wait into smaller increments to maintain responsiveness
                increment = 10  # 10ms increments
                for _ in range(0, remaining_ms, increment):
                    if cv2.waitKey(min(increment, remaining_ms)) & 0xFF == ord('q'):
                        print("Playback interrupted by user")
                        break
                    remaining_ms -= increment
        
        cap.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 