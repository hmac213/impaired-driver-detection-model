import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from filterpy.kalman import KalmanFilter
import torch

class VehicleDetector:
    def __init__(self, model_path='yolov8x.pt', conf_threshold=0.5):
        """Initialize the vehicle detector with YOLO model and tracking capabilities."""
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tracks = defaultdict(list)  # Store tracking history
        self.kalman_filters = {}  # Store Kalman filters for each tracked vehicle
        
        # Vehicle classes we're interested in (from COCO dataset)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
    def init_kalman_filter(self):
        """Initialize Kalman filter for vehicle tracking."""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, dx, dy], Measurement: [x, y]
        kf.F = np.array([[1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        kf.R *= 10  # Measurement noise
        kf.Q *= 0.1  # Process noise
        
        # Initialize state covariance matrix
        kf.P *= 1000
        
        # Initialize state vector
        kf.x = np.zeros((4, 1))
        
        return kf
        
    def detect_and_track(self, frame):
        """Detect vehicles in frame and update their tracks."""
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for det in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det[:6]
            if int(cls) in self.vehicle_classes and conf > self.conf_threshold:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (center_x, center_y),
                    'confidence': conf,
                    'class': int(cls)
                })
                
        # Update tracks with Kalman filtering
        self._update_tracks(detections)
        
        return detections, self.tracks
        
    def _update_tracks(self, detections):
        """Update tracking information for detected vehicles."""
        # If no existing tracks, initialize new ones
        if not self.tracks:
            for i, det in enumerate(detections):
                self.tracks[i].append(det)
                self.kalman_filters[i] = self.init_kalman_filter()
                self.kalman_filters[i].x[:2, 0] = np.array([det['center'][0], det['center'][1]])
            return
            
        # Create cost matrix for matching
        cost_matrix = np.zeros((len(detections), max(len(self.tracks), len(detections))))
        for i, det in enumerate(detections):
            for j in self.tracks.keys():
                if j >= cost_matrix.shape[1]:
                    continue
                last_pos = self.tracks[j][-1]['center']
                cost_matrix[i, j] = np.sqrt(
                    (det['center'][0] - last_pos[0])**2 +
                    (det['center'][1] - last_pos[1])**2
                )
                
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        # Sort detections by confidence
        det_indices = list(range(len(detections)))
        det_indices.sort(key=lambda i: detections[i]['confidence'], reverse=True)
        
        for i in det_indices:
            if i >= len(detections):
                continue
                
            # Find the best matching track for this detection
            min_cost = float('inf')
            best_track = None
            
            for j in self.tracks.keys():
                if j in matched_tracks:
                    continue
                if j >= cost_matrix.shape[1]:
                    continue
                    
                cost = cost_matrix[i, j]
                if cost < min_cost and cost < 100:  # Distance threshold
                    min_cost = cost
                    best_track = j
                    
            if best_track is not None:
                # Update existing track
                self.tracks[best_track].append(detections[i])
                matched_tracks.add(best_track)
                matched_detections.add(i)
                
                # Update Kalman filter
                kf = self.kalman_filters[best_track]
                kf.predict()
                measurement = np.array([[detections[i]['center'][0]], 
                                     [detections[i]['center'][1]]])
                kf.update(measurement)
                
        # Create new tracks for unmatched detections
        for i in range(len(detections)):
            if i not in matched_detections:
                new_id = max(self.tracks.keys(), default=-1) + 1
                self.tracks[new_id].append(detections[i])
                self.kalman_filters[new_id] = self.init_kalman_filter()
                self.kalman_filters[new_id].x[:2, 0] = np.array([detections[i]['center'][0],
                                                                detections[i]['center'][1]])
        
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and tracking information on frame."""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {det['confidence']:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame 