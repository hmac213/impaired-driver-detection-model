import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import platform
from ultralytics import YOLO
import torch

class VehicleDetector:
    """Class for detecting vehicles in images using YOLOv8."""
    
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        # Initialize YOLOv8 model
        self.confidence_threshold = confidence_threshold
        
        # Load YOLOv8 model from Ultralytics
        self.model = YOLO(model_path)
        
        # Vehicle class indices in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO
        
        # Tracking-related attributes
        self.tracks = {}  # Dictionary to store track information
        self.next_track_id = 1  # Counter for assigning track IDs
        
    def detect(self, frame):
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
                    
                    # Calculate center
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    
                    # Get confidence
                    confidence = box.conf.item()
                    
                    # Get class name
                    class_name = result.names[class_id]
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'center': center,
                        'confidence': confidence,
                        'class': class_name
                    }
                    
                    detections.append(detection)
            
        return detections
        
    def detect_and_track(self, frame):
        """Detect vehicles and update tracks."""
        detections = self.detect(frame)
        
        # Simple tracking logic - associate detections with existing tracks
        # based on spatial proximity
        for det in detections:
            matched = False
            
            for track_id, track_data in self.tracks.items():
                if track_data:  # If track has previous detections
                    last_det = track_data[-1]
                    last_center = last_det['center']
                    
                    # Calculate distance between current detection and last point in track
                    dist = np.sqrt(
                        (det['center'][0] - last_center[0])**2 + 
                        (det['center'][1] - last_center[1])**2
                    )
                    
                    # If close enough, associate with this track
                    if dist < 50:  # Threshold for association
                        self.tracks[track_id].append(det)
                        matched = True
                        break
                        
            # If no match found, create new track
            if not matched:
                self.tracks[self.next_track_id] = [det]
                self.next_track_id += 1
                
        return detections, self.tracks

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
    
    # Initialize output video writer if needed
    out = None
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        out = get_video_writer(output_path, fps, frame_width, frame_height)
        if not out or not out.isOpened():
            print(f"Warning: Could not create video writer for {output_path}. Visualization will not be saved.")
            out = None
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing frames")
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect vehicles
        detections, tracks = vehicle_detector.detect_and_track(frame)
        
        # Visualize if requested
        if show_visualization or (output_path and out is not None):
            vis_frame = frame.copy()
            
            # Draw detections
            for det in detections:
                # Draw bounding box
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point
                x, y = det['center']
                cv2.circle(vis_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                
                # Add text with class and confidence
                label = f"{det['class']}: {det['confidence']:.2f}"
                cv2.putText(vis_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
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