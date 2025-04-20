import cv2
import numpy as np
import time
import argparse

class LaneDetector:
    def __init__(self):
        """
        Initialize the lane detector
        """
        # Parameters for lane detection
        self.height_roi = 0.6  # ROI height ratio (lower part of image)
        self.min_line_length = 20
        self.max_line_gap = 20
        
    def detect_lanes(self, frame):
        """
        Detect lanes in a frame
        
        Args:
            frame: Input frame from video
        
        Returns:
            frame_with_lanes: Frame with lane markings
            lane_info: Dictionary containing lane information
        """
        # Make a copy of the frame
        frame_with_lanes = frame.copy()
        height, width = frame.shape[:2]
        
        # Define ROI - bottom portion of the image
        roi_height = int(height * self.height_roi)
        roi = frame[roi_height:height, 0:width]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Apply mask to focus on lane area
        mask = np.zeros_like(edges)
        polygon = np.array([[(0, edges.shape[0]), 
                             (edges.shape[1], edges.shape[0]), 
                             (int(edges.shape[1] * 0.55), int(edges.shape[0] * 0.6)), 
                             (int(edges.shape[1] * 0.45), int(edges.shape[0] * 0.6))]], 
                           np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, 
                               minLineLength=self.min_line_length, 
                               maxLineGap=self.max_line_gap)
        
        # Draw individual detected lane segments
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # offset ROI y-coordinates back to full-frame coordinates
                y1_frame = y1 + roi_height
                y2_frame = y2 + roi_height
                cv2.line(frame_with_lanes, (x1, y1_frame), (x2, y2_frame), (255, 0, 0), 2)

        # Fit and draw converging divider lines per cluster
        if lines is not None:
            # Simple clustering by midpoint x
            clusters = []
            for l in lines:
                mx = (l[0][0] + l[0][2]) / 2
                placed = False
                for c in clusters:
                    if abs(c['center'] - mx) < 50:
                        c['lines'].append(l[0])
                        c['center'] = (c['center'] * (len(c['lines'])-1) + mx) / len(c['lines'])
                        placed = True
                        break
                if not placed:
                    clusters.append({'center': mx, 'lines': [l[0]]})

            for c in clusters:
                pts = []
                for x1, y1, x2, y2 in c['lines']:
                    # shift ROI y back to full-frame
                    y1f, y2f = y1 + roi_height, y2 + roi_height
                    pts.append((x1, y1f))
                    pts.append((x2, y2f))
                # Fit a line y = m*x + b
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                if len(xs) >= 2:
                    m, b = np.polyfit(xs, ys, 1)
                    # Define endpoints at frame bottom and top of ROI
                    y_bottom = height
                    y_top = roi_height
                    x_bottom = int((y_bottom - b) / m)
                    x_top = int((y_top - b) / m)
                    cv2.line(frame_with_lanes, (x_bottom, y_bottom), (x_top, y_top), (0, 255, 0), 2)

        # Initialize lane line arrays
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate slope
                if x2 - x1 == 0:  # Avoid division by zero
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter based on slope: negative slope for left lanes, positive for right lanes
                if slope < -0.3:
                    left_lines.append(line[0])
                elif slope > 0.3:
                    right_lines.append(line[0])
        
        # Combine and extrapolate lane lines
        left_line = self._average_lines(left_lines, height, roi_height)
        right_line = self._average_lines(right_lines, height, roi_height)
        
        lane_info = {
            'left_line': left_line,
            'right_line': right_line
        }
        
        return frame_with_lanes, lane_info
    
    def process_video(self, video_path, output_path=None):
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (if None, won't save)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Output video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Detect lanes
            frame_with_lanes, lane_info = self.detect_lanes(frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps_calc = 10 / (end_time - start_time)
                start_time = end_time
                
                # Display FPS
                cv2.putText(frame_with_lanes, f"FPS: {fps_calc:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the resulting frame
            cv2.imshow('Lane Detection', frame_with_lanes)
            
            # Write frame to output video
            if output_path:
                out.write(frame_with_lanes)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def _average_lines(self, lines, frame_height, roi_height):
        """
        Average and extrapolate lines
        
        Args:
            lines: List of detected lines
            frame_height: Height of the full frame
            roi_height: Height at which ROI starts
            
        Returns:
            Line coordinates [x1, y1, x2, y2] or None if no valid line
        """
        if len(lines) == 0:
            return None
            
        # Average slope and intercept
        slope_sum = 0
        intercept_sum = 0
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            if x2 - x1 == 0:  # Avoid division by zero
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            slope_sum += slope
            intercept_sum += intercept
            
        avg_slope = slope_sum / len(lines)
        avg_intercept = intercept_sum / len(lines)
        
        # Calculate line points for the full frame height
        y1 = frame_height  # Bottom of the image
        y2 = roi_height  # Top of ROI
        
        # Calculate corresponding x values
        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)
        
        return [x1, y1, x2, y2]


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Lane detection")
    parser.add_argument("input_video", help="Path to input video file", nargs="?", default="input_video.mp4")
    parser.add_argument("--output", "-o", help="Path to output video file", default=None)
    
    args = parser.parse_args()
    
    # Initialize the lane detector
    detector = LaneDetector()
    
    print(f"Processing video: {args.input_video}")
    if args.output:
        print(f"Output will be saved to: {args.output}")
    print("Press 'q' to exit")
    
    # Process the video
    detector.process_video(args.input_video, args.output)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()
