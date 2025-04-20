import cv2
import numpy as np
from homography_transformer import HomographyTransformer
from pathlib import Path
import argparse

def test_homography(video_path, output_path=None, calibration_path=None):
    """
    Test the homography transformation on a video.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the output video
        calibration_path: Path to the homography calibration file
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read first frame for calibration
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    
    # Initialize homography transformer
    homography_transformer = HomographyTransformer()
    
    # Load or create calibration
    if calibration_path and Path(calibration_path).exists():
        print(f"Loading calibration from {calibration_path}")
        homography_transformer.load_calibration(calibration_path)
    else:
        print("Please calibrate the homography transformation...")
        print("1. Select 4 points: bottom-left, bottom-right, top-right, top-left")
        print("2. After selecting points: Press Enter to adjust the rectangle or 'a' to accept as is")
        print("3. In adjustment mode: Drag points to fine-tune, 'r' to reset, Enter to confirm")
        homography_transformer.calibrate(first_frame)
        
        # Save calibration if a path is provided
        if calibration_path:
            homography_transformer.save_calibration(calibration_path)
            print(f"Calibration saved to {calibration_path}")
    
    # Create output video writer if needed
    out = None
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            fps, 
            (frame_width + homography_transformer.warped_size[0], frame_height)
        )
        
        if not out or not out.isOpened():
            print(f"Warning: Could not create video writer for {output_path}. Visualization will not be saved.")
            out = None
    
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("Processing video... Press 'q' to quit.")
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply homography transformation
        warped_frame = homography_transformer.warp_frame(frame)
        
        # Create a side-by-side display
        combined_frame = np.zeros((frame_height, frame_width + warped_frame.shape[1], 3), dtype=np.uint8)
        combined_frame[:, :frame_width] = frame
        
        # If warped frame is smaller than original frame height, center it vertically
        if warped_frame.shape[0] < frame_height:
            y_offset = (frame_height - warped_frame.shape[0]) // 2
            combined_frame[y_offset:y_offset + warped_frame.shape[0], frame_width:frame_width + warped_frame.shape[1]] = warped_frame
        else:
            # If warped frame is taller, resize it to match frame_height
            resized_warped = cv2.resize(warped_frame, (warped_frame.shape[1], frame_height))
            combined_frame[:, frame_width:frame_width + warped_frame.shape[1]] = resized_warped
        
        # Add text labels
        cv2.putText(combined_frame, "Original View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_frame, "Bird's-eye View", (frame_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw separator line
        cv2.line(combined_frame, (frame_width, 0), (frame_width, frame_height), (255, 255, 255), 2)
        
        # Show result
        cv2.imshow("Homography Test", combined_frame)
        
        # Write frame to output video
        if out is not None:
            out.write(combined_frame)
        
        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print("Processing complete.")

def main():
    parser = argparse.ArgumentParser(description='Test homography transformation on a video')
    parser.add_argument('video_path', help='Path to input video file', nargs='?', default='test.mp4')
    parser.add_argument('--output', help='Path to output video file')
    parser.add_argument('--calibration', help='Path to homography calibration file')
    args = parser.parse_args()
    
    # Convert paths to Path objects
    video_path = Path(args.video_path)
    output_path = Path(args.output) if args.output else None
    calibration_path = Path(args.calibration) if args.calibration else None
    
    # Verify input file exists
    if not video_path.exists():
        print(f"Input video file not found: {video_path}")
        return
    
    test_homography(video_path, output_path, calibration_path)

if __name__ == '__main__':
    main() 