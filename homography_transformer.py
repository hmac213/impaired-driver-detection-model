import cv2
import numpy as np
from pathlib import Path

class HomographyTransformer:
    def __init__(self, frame=None):
        """
        Initialize the homography transformer for road flattening.
        
        Args:
            frame: Optional frame to use for calibration
        """
        self.homography_matrix = None
        self.inverse_homography_matrix = None
        self.warped_size = None
        self.source_points = None
        self.destination_points = None
        
        # If a frame is provided, use it for calibration
        if frame is not None:
            self.calibrate(frame)
            
    def calibrate(self, frame):
        """
        Calibrate the homography transformation by selecting points.
        
        Args:
            frame: Frame to use for calibration
        """
        self.points = []
        window_name = "Homography Calibration"
        
        # Make a copy of the frame for visualization
        display_frame = frame.copy()
        
        # Define callback function for mouse events
        def select_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add the point to the list
                self.points.append((x, y))
                # Draw the point on the image
                cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow(window_name, display_frame)
                
                if len(self.points) == 4:
                    print("4 points selected. Press 'Enter' to continue to rectangle adjustment or 'a' to accept points as is.")
        
        # Create a window and set the callback
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_points)
        
        print("Select 4 points in the image to define the road region (bottom-left, bottom-right, top-right, top-left).")
        cv2.imshow(window_name, display_frame)
        
        # Wait for user to select points and press Enter
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(self.points) == 4:  # Enter key and 4 points selected
                break
            elif key == ord('a') and len(self.points) == 4:  # 'a' key to accept as is
                cv2.destroyWindow(window_name)
                if self.points is None or len(self.points) != 4:
                    raise ValueError("Calibration failed. Need exactly 4 points.")
                
                # Sort points to ensure they are in the correct order: bottom-left, bottom-right, top-right, top-left
                self.source_points = np.array(self.points, dtype=np.float32)
                self._calculate_homography()
                return True
            elif key == 27:  # ESC key
                self.points = None
                break
        
        # If we got here, user wants to adjust the rectangle
        if len(self.points) == 4:
            self._adjust_rectangle(frame, window_name)
        else:
            cv2.destroyWindow(window_name)
            raise ValueError("Calibration failed. Need exactly 4 points.")
        
        return True
    
    def _adjust_rectangle(self, frame, window_name):
        """
        Allow user to adjust the rectangle by dragging corners.
        
        Args:
            frame: Original frame
            window_name: Window name for display
        """
        adjustment_frame = frame.copy()
        points = np.array(self.points, dtype=np.int32)
        
        # Draw initial rectangle
        cv2.polylines(adjustment_frame, [points], True, (0, 255, 0), 2)
        for i, pt in enumerate(points):
            cv2.circle(adjustment_frame, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(adjustment_frame, str(i), (pt[0]+10, pt[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, adjustment_frame)
        print("Adjust rectangle: Drag points to adjust, press 'Enter' to confirm, 'r' to reset.")
        
        # Variables for point dragging
        dragging = False
        drag_point_idx = -1
        drag_threshold = 20  # pixels
        
        def mouse_adjust(event, x, y, flags, param):
            nonlocal dragging, drag_point_idx, adjustment_frame
            
            # Check if we're near a point for dragging
            if event == cv2.EVENT_LBUTTONDOWN:
                for i, pt in enumerate(points):
                    if np.sqrt((x - pt[0])**2 + (y - pt[1])**2) < drag_threshold:
                        dragging = True
                        drag_point_idx = i
                        break
            
            # Update point position while dragging
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                points[drag_point_idx] = (x, y)
                # Redraw the rectangle
                adjustment_frame = frame.copy()
                cv2.polylines(adjustment_frame, [points], True, (0, 255, 0), 2)
                for i, pt in enumerate(points):
                    color = (0, 0, 255) if i == drag_point_idx else (0, 0, 255)
                    cv2.circle(adjustment_frame, tuple(pt), 5, color, -1)
                    cv2.putText(adjustment_frame, str(i), (pt[0]+10, pt[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(window_name, adjustment_frame)
            
            # End dragging
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False
                drag_point_idx = -1
        
        # Set mouse callback for adjustment
        cv2.setMouseCallback(window_name, mouse_adjust)
        
        # Wait for user confirmation
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key to confirm
                break
            elif key == ord('r'):  # Reset to original points
                points = np.array(self.points, dtype=np.int32)
                adjustment_frame = frame.copy()
                cv2.polylines(adjustment_frame, [points], True, (0, 255, 0), 2)
                for i, pt in enumerate(points):
                    cv2.circle(adjustment_frame, tuple(pt), 5, (0, 0, 255), -1)
                    cv2.putText(adjustment_frame, str(i), (pt[0]+10, pt[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(window_name, adjustment_frame)
            elif key == 27:  # ESC key to cancel
                points = np.array(self.points, dtype=np.int32)
                break
        
        cv2.destroyWindow(window_name)
        
        # Update the points with adjusted ones
        self.points = [tuple(pt) for pt in points]
        
        # Convert to correct format and calculate homography
        self.source_points = np.array(self.points, dtype=np.float32)
        self._calculate_homography()
    
    def _calculate_homography(self):
        """Calculate homography matrix from source points."""
        # Calculate width and height for the bird's-eye view
        width_bottom = np.sqrt(((self.source_points[1][0] - self.source_points[0][0]) ** 2) + 
                              ((self.source_points[1][1] - self.source_points[0][1]) ** 2))
        width_top = np.sqrt(((self.source_points[2][0] - self.source_points[3][0]) ** 2) + 
                           ((self.source_points[2][1] - self.source_points[3][1]) ** 2))
        height_left = np.sqrt(((self.source_points[3][0] - self.source_points[0][0]) ** 2) + 
                             ((self.source_points[3][1] - self.source_points[0][1]) ** 2))
        height_right = np.sqrt(((self.source_points[2][0] - self.source_points[1][0]) ** 2) + 
                              ((self.source_points[2][1] - self.source_points[1][1]) ** 2))
        
        # Take the maximum dimensions
        max_width = max(int(width_bottom), int(width_top))
        max_height = max(int(height_left), int(height_right))
        
        # Define the destination points for the warped image (rectangle)
        self.destination_points = np.array([
            [0, max_height - 1],               # bottom-left
            [max_width - 1, max_height - 1],   # bottom-right
            [max_width - 1, 0],                # top-right
            [0, 0]                             # top-left
        ], dtype=np.float32)
        
        self.warped_size = (max_width, max_height)
        
        # Calculate the homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(self.source_points, self.destination_points)
        self.inverse_homography_matrix = cv2.getPerspectiveTransform(self.destination_points, self.source_points)
        
    def warp_frame(self, frame):
        """
        Apply perspective transformation to a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Warped frame (bird's-eye view)
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not calculated. Call calibrate() first.")
            
        return cv2.warpPerspective(frame, self.homography_matrix, self.warped_size)
    
    def warp_point(self, point):
        """
        Apply perspective transformation to a single point.
        
        Args:
            point: (x, y) coordinate in original frame
            
        Returns:
            (x, y) coordinate in warped frame
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not calculated. Call calibrate() first.")
            
        # Convert point to homogeneous coordinates
        point_h = np.array([point[0], point[1], 1], dtype=np.float32)
        
        # Apply transformation
        warped_point_h = np.dot(self.homography_matrix, point_h)
        
        # Convert back to cartesian coordinates
        warped_point = (int(warped_point_h[0] / warped_point_h[2]), 
                        int(warped_point_h[1] / warped_point_h[2]))
        
        return warped_point
    
    def unwarp_point(self, point):
        """
        Apply inverse perspective transformation to a single point.
        
        Args:
            point: (x, y) coordinate in warped frame
            
        Returns:
            (x, y) coordinate in original frame
        """
        if self.inverse_homography_matrix is None:
            raise ValueError("Inverse homography matrix not calculated. Call calibrate() first.")
            
        # Convert point to homogeneous coordinates
        point_h = np.array([point[0], point[1], 1], dtype=np.float32)
        
        # Apply transformation
        original_point_h = np.dot(self.inverse_homography_matrix, point_h)
        
        # Convert back to cartesian coordinates
        original_point = (int(original_point_h[0] / original_point_h[2]), 
                          int(original_point_h[1] / original_point_h[2]))
        
        return original_point
    
    def save_calibration(self, file_path):
        """
        Save the calibration parameters to a file.
        
        Args:
            file_path: Path to save the calibration file
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not calculated. Call calibrate() first.")
            
        calibration_data = {
            "source_points": self.source_points.tolist(),
            "destination_points": self.destination_points.tolist(),
            "warped_size": self.warped_size
        }
        
        np.save(file_path, calibration_data)
        
    def load_calibration(self, file_path):
        """
        Load calibration parameters from a file.
        
        Args:
            file_path: Path to the calibration file
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Calibration file not found: {file_path}")
            
        calibration_data = np.load(file_path, allow_pickle=True).item()
        
        self.source_points = np.array(calibration_data["source_points"], dtype=np.float32)
        self.destination_points = np.array(calibration_data["destination_points"], dtype=np.float32)
        self.warped_size = calibration_data["warped_size"]
        
        # Recalculate homography matrices
        self.homography_matrix = cv2.getPerspectiveTransform(self.source_points, self.destination_points)
        self.inverse_homography_matrix = cv2.getPerspectiveTransform(self.destination_points, self.source_points)
        
        return True 