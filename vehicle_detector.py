import cv2
import numpy as np
import os

class VehicleDetector:
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the vehicle detector with background subtraction
        
        Args:
            confidence_threshold: Detection confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize vehicle tracking
        self.next_id = 1
        self.tracks = {}  # Dictionary to store tracking info
        self.max_disappeared = 15  # Maximum frames to keep tracking a disappeared object
        self.disappeared = {}  # Count of frames an object has been missing
        
        # Vehicle classes in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30, detectShadows=True)
        
        # Keep track of frame history for detection stability
        self.frame_history = []
        self.history_size = 5
        
        # Minimum contour area to be considered a vehicle (adjust based on video resolution)
        self.min_contour_area = 400
        
        # Frame counter for periodic learning rate adjustment
        self.frame_count = 0
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in a frame using background subtraction
        
        Args:
            frame: Input frame
            
        Returns:
            detections: List of dictionaries with detection info
        """
        # Increment frame counter
        self.frame_count += 1
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply background subtraction with shadow detection
        fg_mask = self.bg_subtractor.apply(blurred, learningRate=0.002)
        
        # Remove shadows (value 127) and keep only foreground (value 255)
        foreground_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Apply morphological operations to remove noise and fill gaps
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((11, 11), np.uint8)
        
        # Opening (erosion followed by dilation) - removes small noise
        opening = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Closing (dilation followed by erosion) - fills small holes
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # Additional dilation to merge nearby contours
        dilation = cv2.dilate(closing, kernel_open, iterations=1)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create region of interest mask for the road area
        # This helps exclude detections in irrelevant areas
        roi_mask = np.zeros_like(dilation)
        # Define polygon for the road area - adjust these points based on your video
        road_polygon = np.array([
            [(0, height), (width, height), (width*0.65, height*0.65), (width*0.35, height*0.65)]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, road_polygon, 255)
        
        # Process contours and create detections
        detections = []
        for contour in contours:
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
                
            # Get bounding box of contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create detection with center point
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Check if the center of the contour is within the road area
            if roi_mask[center_y, center_x] == 0:
                continue
                
            # Filter boxes by aspect ratio (to exclude non-vehicle objects)
            aspect_ratio = float(w) / h
            if not (0.5 <= aspect_ratio <= 3.0):
                continue
                
            # Filter by size relative to frame
            relative_area = area / (height * width)
            if relative_area < 0.0005 or relative_area > 0.05:
                continue
            
            # Assign a vehicle class based on the aspect ratio and size
            if aspect_ratio > 1.5:
                class_idx = 2  # car
            elif aspect_ratio > 0.8:
                class_idx = 5 if area > 5000 else 2  # bus if large, car if smaller
            else:
                class_idx = 3  # motorcycle
                
            class_name = self.classes[class_idx] if class_idx < len(self.classes) else "unknown"
            
            # Calculate confidence based on area and relative position
            confidence = min(1.0, area / 5000)
            
            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': confidence,
                'class': class_idx,
                'class_name': class_name,
                'center': (center_x, center_y)
            })
        
        # Add current detections to history
        self.frame_history.append(detections)
        if len(self.frame_history) > self.history_size:
            self.frame_history.pop(0)
        
        # Stabilize detections by considering detection history
        if len(self.frame_history) >= 2:
            # Use the current detections as the base
            stable_detections = detections.copy()
            
            # Occasionally reset the background model to adapt to changes
            if self.frame_count % 500 == 0:
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=200, varThreshold=30, detectShadows=True)
                
        else:
            stable_detections = detections
        
        return stable_detections
    
    def assign_tracks(self, detections):
        """
        Assign track IDs to detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            tracks: Dictionary of tracks updated
        """
        # If no existing tracks, create new tracks for all detections
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = [det]
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return self.tracks
            
        # If no current detections, increment disappeared count for all tracks
        if not detections:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Remove track if disappeared for too many frames
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.tracks[object_id]
                    del self.disappeared[object_id]
            return self.tracks
        
        # Calculate distances between existing tracks and new detections
        distances = {}
        for track_id, track_data in self.tracks.items():
            if not track_data:
                continue
                
            last_position = track_data[-1]['center']
            
            for i, det in enumerate(detections):
                center = det['center']
                dist = np.sqrt((last_position[0] - center[0])**2 + (last_position[1] - center[1])**2)
                distances[(track_id, i)] = dist
        
        # Sort distances and assign detections to existing tracks
        used_detections = set()
        updated_tracks = set()
        
        if distances:
            sorted_distances = sorted(distances.items(), key=lambda x: x[1])
            
            for (track_id, det_idx), dist in sorted_distances:
                if det_idx not in used_detections and track_id not in updated_tracks and dist < 100:
                    self.tracks[track_id].append(detections[det_idx])
                    self.disappeared[track_id] = 0
                    used_detections.add(det_idx)
                    updated_tracks.add(track_id)
        
        # Create new tracks for unassigned detections
        for i, det in enumerate(detections):
            if i not in used_detections:
                self.tracks[self.next_id] = [det]
                self.disappeared[self.next_id] = 0
                self.next_id += 1
                
        # Increment disappeared count for tracks with no match
        for track_id in list(self.tracks.keys()):
            if track_id not in updated_tracks:
                self.disappeared[track_id] += 1
                
                # Remove track if disappeared for too many frames
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
                    
        return self.tracks
    
    def detect_and_track(self, frame):
        """
        Detect and track vehicles in a frame
        
        Args:
            frame: Input frame
            
        Returns:
            detections: List of detection dictionaries
            tracks: Dictionary of tracks
        """
        detections = self.detect_vehicles(frame)
        tracks = self.assign_tracks(detections)
        return detections, tracks

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and tracking information on frame."""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Different colors for different vehicle types
            if det['class'] == 2:  # car
                color = (0, 255, 0)  # green
            elif det['class'] == 3:  # motorcycle
                color = (255, 0, 0)  # blue
            elif det['class'] == 5:  # bus
                color = (0, 0, 255)  # red
            elif det['class'] == 7:  # truck
                color = (255, 255, 0)  # cyan
            else:
                color = (255, 0, 255)  # magenta
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{det['class_name']}: {det['confidence']:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame 