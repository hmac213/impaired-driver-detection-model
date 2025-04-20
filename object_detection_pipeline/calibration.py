import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Tuple

class LaneDetector:
    """Class for detecting and managing lane boundaries."""
    
    def __init__(self):
        self.lanes = []  # List of lane functions
        self.points = []  # List of points for each lane
        self.homography_matrix = None
        self.rectangle_points = []  # Points for homography rectangle
        self.world_width = 3.7  # Standard lane width in meters
        self.world_length = 30.0  # Length of the ROI in meters
        
    def get_homography_rectangle(self, frame: np.ndarray) -> None:
        """Get rectangle points from user for homography calculation."""
        temp_frame = frame.copy()
        window_name = 'Select ROI Rectangle'
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.rectangle_points) < 4:
                    self.rectangle_points.append((x, y))
                    # Draw the new point
                    cv2.circle(temp_frame, (x, y), 5, (0, 255, 0), -1)
                    # Draw line to previous point if it exists
                    if len(self.rectangle_points) > 1:
                        cv2.line(temp_frame, self.rectangle_points[-2], (x, y), (0, 255, 0), 2)
                    # Draw closing line if we have all points
                    if len(self.rectangle_points) == 4:
                        cv2.line(temp_frame, self.rectangle_points[-1], self.rectangle_points[0], (0, 255, 0), 2)
                    cv2.imshow(window_name, temp_frame)
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        print("Select 4 points to define the ROI rectangle (press 'q' when done)")
        
        while True:
            cv2.imshow(window_name, temp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or len(self.rectangle_points) == 4:
                break
        
        if len(self.rectangle_points) == 4:
            # Let user select entrance and exit sides
            print("\nClick on the entrance side of the rectangle (where vehicles enter)")
            print("Then click on the exit side (where vehicles leave)")
            
            entrance_side = None
            exit_side = None
            
            def side_callback(event, x, y, flags, param):
                nonlocal entrance_side, exit_side
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Find which side was clicked
                    for i in range(4):
                        p1 = self.rectangle_points[i]
                        p2 = self.rectangle_points[(i + 1) % 4]
                        # Check if click is near this side
                        if self._is_point_near_line((x, y), p1, p2):
                            if entrance_side is None:
                                entrance_side = i
                                print(f"Entrance side selected: {i}")
                                # Draw entrance side in blue
                                cv2.line(temp_frame, p1, p2, (255, 0, 0), 2)
                                cv2.imshow(window_name, temp_frame)
                            elif exit_side is None and i != entrance_side:
                                exit_side = i
                                print(f"Exit side selected: {i}")
                                # Draw exit side in red
                                cv2.line(temp_frame, p1, p2, (0, 0, 255), 2)
                                cv2.imshow(window_name, temp_frame)
                            break
            
            cv2.setMouseCallback(window_name, side_callback)
            
            while entrance_side is None or exit_side is None:
                cv2.imshow(window_name, temp_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Store the entrance and exit sides
            self.entrance_side = entrance_side
            self.exit_side = exit_side
            
            # Calculate homography matrix
            src_points = np.float32(self.rectangle_points)
            
            # Define destination points in world coordinates (meters)
            # The x-axis will be along the entrance-to-exit direction
            dst_points = np.float32([
                [0, 0],  # top-left
                [self.world_width, 0],  # top-right
                [self.world_width, self.world_length],  # bottom-right
                [0, self.world_length]  # bottom-left
            ])
            
            self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            self.inv_homography_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
            
            print("ROI rectangle selected successfully!")
        else:
            print("Warning: ROI rectangle selection incomplete. Using default homography.")
        
        cv2.destroyAllWindows()
        
    def _is_point_near_line(self, point, line_start, line_end, threshold=10):
        """Check if a point is near a line segment."""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate distance from point to line segment
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return False
            
        # Calculate the projection of the point onto the line
        t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_length**2)
        t = max(0, min(1, t))  # Clamp to line segment
        
        # Find the closest point on the line
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)
        
        # Calculate distance from point to closest point on line
        distance = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
        
        return distance <= threshold
        
    def get_lane_points(self, frame: np.ndarray) -> None:
        """Get lane points from user input."""
        num_lanes = int(input("Enter the number of lane lines to draw: "))
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Select Lane Points', frame)
        
        cv2.namedWindow('Select Lane Points')
        cv2.setMouseCallback('Select Lane Points', mouse_callback)
        
        for lane_idx in range(num_lanes):
            print(f"Select points for lane {lane_idx + 1} (press 'q' when done)")
            self.points = []
            while True:
                cv2.imshow('Select Lane Points', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # Fit polynomial to points
            if len(self.points) >= 2:
                points = np.array(self.points)
                coeffs = np.polyfit(points[:, 1], points[:, 0], 2)  # x = f(y)
                self.lanes.append(coeffs)
        
        cv2.destroyAllWindows()
        
    def get_lane_position(self, point: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate normalized position (-1 to 1) of a point relative to lane boundaries and distance along lane."""
        if not self.lanes or len(self.lanes) < 2 or self.homography_matrix is None:
            return 0.0, 0.0
            
        # Convert point to world coordinates
        point_homogeneous = np.array([point[0], point[1], 1])
        world_point = np.dot(self.homography_matrix, point_homogeneous)
        world_point = world_point[:2] / world_point[2]  # Convert from homogeneous coordinates
        
        # Find the two nearest lane lines
        y = point[1]
        lane_positions = []
        for coeffs in self.lanes:
            lane_x = np.polyval(coeffs, y)
            lane_positions.append(lane_x)
        
        lane_positions.sort()
        left_lane = None
        right_lane = None
        
        for i in range(len(lane_positions) - 1):
            if lane_positions[i] <= point[0] <= lane_positions[i + 1]:
                left_lane = lane_positions[i]
                right_lane = lane_positions[i + 1]
                break
        
        if left_lane is None or right_lane is None:
            return 0.0, 0.0
        
        # Calculate normalized position
        lane_width = right_lane - left_lane
        position = (point[0] - left_lane) / lane_width * 2 - 1  # Normalize to [-1, 1]
        
        # Distance along lane is the y-coordinate in world space
        distance = world_point[1]
        
        return position, distance

    def is_point_in_rectangle(self, point: Tuple[int, int]) -> Tuple[bool, float]:
        """Check if a point is within the rectangle and return its relative x-position (0-1).
        Returns (False, 0.0) if point is outside the rectangle or if rectangle is not defined."""
        if len(self.rectangle_points) != 4 or self.homography_matrix is None:
            return False, 0.0
            
        # Convert point to homogeneous coordinates
        point_homogeneous = np.array([point[0], point[1], 1])
        
        # Transform point to world coordinates using homography
        world_point = np.dot(self.homography_matrix, point_homogeneous)
        world_point = world_point[:2] / world_point[2]  # Convert from homogeneous coordinates
        
        # Get the entrance and exit points in world coordinates
        entrance_p1 = np.array([0, 0])
        entrance_p2 = np.array([self.world_width, 0])
        exit_p1 = np.array([0, self.world_length])
        exit_p2 = np.array([self.world_width, self.world_length])
        
        # Calculate the distance from the point to the entrance and exit sides
        def distance_to_line(point, line_p1, line_p2):
            line_vec = line_p2 - line_p1
            point_vec = point - line_p1
            line_length = np.linalg.norm(line_vec)
            if line_length == 0:
                return np.inf
            return np.abs(np.cross(line_vec, point_vec)) / line_length
        
        entrance_dist = distance_to_line(world_point, entrance_p1, entrance_p2)
        exit_dist = distance_to_line(world_point, exit_p1, exit_p2)
        
        # Check if point is within the rectangle's bounds in world coordinates
        is_inside = (0 <= world_point[0] <= self.world_width and 
                    0 <= world_point[1] <= self.world_length)
        
        # Calculate relative x-position based on distance from entrance
        total_length = self.world_length
        relative_x = world_point[1] / total_length  # y-coordinate in world space represents progress
        relative_x = max(0.0, min(1.0, relative_x))  # Clamp between 0 and 1
        
        return is_inside, relative_x

class VehicleDetector:
    """Class for detecting vehicles in images using YOLOv8."""
    
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        # Initialize YOLOv8 model
        self.confidence_threshold = confidence_threshold
        
        # Load YOLOv8 model from Ultralytics
        self.model = YOLO(model_path)
        
        # Vehicle class indices in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO
        
        # Reference point for vehicle position
        self.ref_point = None  # Will store (x_offset, y_offset) relative to bbox
        
    def get_reference_point(self, frame: np.ndarray, fully_visible_detection: dict) -> None:
        """Let user select a reference point on a sample vehicle detection."""
        if fully_visible_detection is None:
            print("No fully visible vehicle detected. Using default bottom center.")
            self.ref_point = (0.5, 1.0)  # Default to bottom center
            return
            
        # Use the fully visible detection
        x1, y1, x2, y2 = fully_visible_detection['bbox']
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        temp_frame = frame.copy()
        
        # Draw only the fully visible vehicle's bounding box
        cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Create a window and set mouse callback
        window_name = 'Select Reference Point'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Store the reference point and bbox coordinates for the callback
        class CallbackData:
            def __init__(self):
                self.ref_point = None
                self.bbox = (x1, y1, x2, y2)
                self.frame = temp_frame.copy()
        
        callback_data = CallbackData()
        
        def mouse_callback(event, x, y, flags, param):
            data = param
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if click is within the bounding box
                if (data.bbox[0] <= x <= data.bbox[2] and 
                    data.bbox[1] <= y <= data.bbox[3]):
                    # Calculate relative position within bbox
                    rel_x = (x - data.bbox[0]) / (data.bbox[2] - data.bbox[0])
                    rel_y = (y - data.bbox[1]) / (data.bbox[3] - data.bbox[1])
                    
                    # Store the reference point
                    data.ref_point = (rel_x, rel_y)
                    
                    # Draw the selected point
                    vis_frame = data.frame.copy()
                    cv2.circle(vis_frame, (x, y), 5, (0, 0, 255), -1)
                    cv2.imshow(window_name, vis_frame)
        
        cv2.setMouseCallback(window_name, mouse_callback, callback_data)
        
        print("\nClick inside the green bounding box to select the reference point")
        print("Press 'q' when done")
        
        while True:
            cv2.imshow(window_name, callback_data.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or callback_data.ref_point is not None:
                break
        
        cv2.destroyAllWindows()
        
        if callback_data.ref_point is not None:
            self.ref_point = callback_data.ref_point
            print(f"Reference point selected at relative position: {self.ref_point}")
        else:
            print("No reference point selected. Using default bottom center.")
            self.ref_point = (0.5, 1.0)  # Default to bottom center

    def detect(self, frame: np.ndarray) -> list:
        """Detect vehicles in a frame."""
        # Run YOLOv8 inference on the frame
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Prepare detections
        detections = []
        
        # Process results (get first result as we only processed one image)
        if results and len(results) > 0:
            result = results[0]
            
            # Get boxes, confidences and class_ids
            for box in result.boxes:
                class_id = int(box.cls.item())
                
                # Filter for vehicle classes
                if class_id in self.vehicle_classes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Calculate reference point position
                    if self.ref_point is None:
                        self.ref_point = (0.5, 1.0)  # Default to bottom center
                    
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    ref_x = x1 + int(self.ref_point[0] * bbox_width)
                    ref_y = y1 + int(self.ref_point[1] * bbox_height)
                    ref_point = (ref_x, ref_y)
                    
                    # Get confidence
                    confidence = box.conf.item()
                    
                    # Get class name
                    class_name = result.names[class_id]
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'center': ref_point,  # Using the reference point
                        'confidence': confidence,
                        'class': class_name
                    }
                    
                    detections.append(detection)
            
        return detections

def setup_processing(video_path: str) -> Tuple[VehicleDetector, LaneDetector, dict]:
    """Perform all setup steps and return the configured detectors and video info."""
    # Initialize detectors
    vehicle_detector = VehicleDetector()
    lane_detector = LaneDetector()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    # Get video properties
    video_info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'frame_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    
    # Scan for a frame with vehicles fully in view
    print("Scanning for a frame with vehicles fully in view...")
    setup_frame = None
    setup_frame_idx = 0
    margin = 100  # Increased margin to ensure vehicle is well within frame
    fully_visible_detection = None
    
    def is_fully_in_view(bbox, frame_width, frame_height, margin):
        x1, y1, x2, y2 = bbox
        # Check if bounding box is well within frame margins
        if not (x1 > margin and y1 > margin and 
                x2 < frame_width - margin and y2 < frame_height - margin):
            return False
            
        # Additional checks for vehicle visibility
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Check if bounding box has reasonable proportions (not too narrow or too wide)
        aspect_ratio = bbox_width / bbox_height
        if not (0.5 < aspect_ratio < 3.0):  # Typical vehicle aspect ratios
            return False
            
        # Check if bounding box is large enough to be a complete vehicle
        min_size = 100  # Minimum size in pixels
        if bbox_width < min_size or bbox_height < min_size:
            return False
            
        return True
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Try to detect vehicles
        detections = vehicle_detector.detect(frame)
        if detections:
            # Check if any vehicle is fully in view
            for det in detections:
                if is_fully_in_view(det['bbox'], video_info['frame_width'], video_info['frame_height'], margin):
                    setup_frame = frame
                    fully_visible_detection = det
                    print(f"Found suitable vehicle at frame {setup_frame_idx}")
                    print(f"Vehicle size: {det['bbox'][2]-det['bbox'][0]}x{det['bbox'][3]-det['bbox'][1]} pixels")
                    break
            
            if setup_frame is not None:
                break
            
        setup_frame_idx += 1
        if setup_frame_idx % 30 == 0:  # Print progress every second (assuming 30fps)
            print(f"Scanned {setup_frame_idx} frames...")
    
    if setup_frame is None:
        raise ValueError("No suitable vehicles fully in view detected in the video. Please try a different video.")
    
    print(f"Found a frame with a suitable vehicle fully in view at frame {setup_frame_idx}")
    
    # Get reference point for vehicle position
    print("\nSelect the reference point for vehicle position:")
    print("Click anywhere within the bounding box of the vehicle")
    print("This point will be used for all vehicle detections")
    vehicle_detector.get_reference_point(setup_frame, fully_visible_detection)
    
    # Get homography rectangle and lane points from user
    print("\nNow select the ROI rectangle for homography:")
    lane_detector.get_homography_rectangle(setup_frame)
    
    print("\nNow select the lane lines:")
    lane_detector.get_lane_points(setup_frame)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    return vehicle_detector, lane_detector, video_info 