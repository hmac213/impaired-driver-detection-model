import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def preprocess_video(input_path: str, output_path: str = None) -> str:
    """
    Preprocess video by converting it to 10 fps.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save processed video (optional)
        
    Returns:
        Path to the processed video file
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If no output path provided, create one with _10fps suffix
    if output_path is None:
        input_path = Path(input_path)
        output_path = str(input_path.parent / f"{input_path.stem}_10fps{input_path.suffix}")
    
    # Create video writer for 10 fps output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10, (width, height))
    
    # Calculate frame interval to achieve 10 fps
    frame_interval = int(fps / 10)
    
    print(f"Converting video to 10 fps (original fps: {fps})")
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Write every nth frame to achieve 10 fps
        if frame_count % frame_interval == 0:
            out.write(frame)
            
        frame_count += 1
        pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    pbar.close()
    
    print(f"Processed video saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert video to 10 fps")
    parser.add_argument("input_path", help="Path to input video file")
    parser.add_argument("--output", help="Path to output video file (optional)")
    args = parser.parse_args()
    
    preprocess_video(args.input_path, args.output) 