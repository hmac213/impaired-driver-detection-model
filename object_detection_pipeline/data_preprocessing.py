import pandas as pd
from pathlib import Path
import argparse
import numpy as np
from scipy.interpolate import interp1d

def interpolate_gaps(track_data: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """
    Interpolate gaps in track data up to max_gap frames.
    
    Args:
        track_data: DataFrame containing track data for a single vehicle
        max_gap: Maximum number of frames to interpolate between
        
    Returns:
        DataFrame with gaps interpolated
    """
    if len(track_data) <= 1:
        return track_data
        
    # Create a complete frame sequence from min to max frame
    min_frame = track_data['frame'].min()
    max_frame = track_data['frame'].max()
    all_frames = np.arange(min_frame, max_frame + 1)
    
    # Find missing frames
    existing_frames = track_data['frame'].values
    missing_frames = np.setdiff1d(all_frames, existing_frames)
    
    # Group missing frames into gaps
    gaps = []
    current_gap = []
    
    for frame in missing_frames:
        if not current_gap or frame == current_gap[-1] + 1:
            current_gap.append(frame)
        else:
            if current_gap:
                gaps.append(current_gap)
            current_gap = [frame]
    if current_gap:
        gaps.append(current_gap)
    
    # Interpolate gaps that are within max_gap size
    interpolated_data = track_data.copy()
    
    for gap in gaps:
        if len(gap) > max_gap:
            continue
            
        # Get frames before and after the gap
        before_frame = gap[0] - 1
        after_frame = gap[-1] + 1
        
        # Get data points before and after the gap
        before_data = track_data[track_data['frame'] == before_frame]
        after_data = track_data[track_data['frame'] == after_frame]
        
        if len(before_data) == 0 or len(after_data) == 0:
            continue
            
        # Create interpolation functions for each column
        x_interp = interp1d([before_frame, after_frame], 
                           [before_data['x'].iloc[0], after_data['x'].iloc[0]])
        y_interp = interp1d([before_frame, after_frame], 
                           [before_data['y'].iloc[0], after_data['y'].iloc[0]])
        time_interp = interp1d([before_frame, after_frame], 
                             [before_data['time'].iloc[0], after_data['time'].iloc[0]])
        
        # Create new rows for the gap
        for frame in gap:
            new_row = {
                'track_id': track_data['track_id'].iloc[0],
                'frame': frame,
                'time': float(time_interp(frame)),
                'x': float(x_interp(frame)),
                'y': float(y_interp(frame)),
                'confidence': (before_data['confidence'].iloc[0] + after_data['confidence'].iloc[0]) / 2,
                'class': before_data['class'].iloc[0]
            }
            interpolated_data = pd.concat([interpolated_data, pd.DataFrame([new_row])], ignore_index=True)
    
    # Sort by frame and reset index
    interpolated_data = interpolated_data.sort_values('frame').reset_index(drop=True)
    return interpolated_data

def process_track_data(csv_path: str, max_gap: int = 3):
    """
    Process track data CSV file and create separate CSV files for each vehicle track.
    Discards frames with multiple detections and interpolates gaps up to max_gap frames.
    
    Args:
        csv_path: Path to the input CSV file containing track data
        max_gap: Maximum number of frames to interpolate between
    """
    # Read the input CSV file
    df = pd.read_csv(csv_path)
    
    # Get the output directory from the input file path
    input_path = Path(csv_path)
    output_dir = input_path.parent / "vehicle_tracks"
    output_dir.mkdir(exist_ok=True)
    
    # Get unique track IDs
    track_ids = df['track_id'].unique()
    
    print(f"Processing {len(track_ids)} vehicle tracks...")
    
    # Process each track
    for track_id in track_ids:
        # Filter data for this track
        track_data = df[df['track_id'] == track_id].copy()
        
        # Group by frame and count detections
        frame_counts = track_data.groupby('frame').size()
        
        # Find frames with multiple detections
        multiple_detection_frames = frame_counts[frame_counts > 1].index
        
        # Remove frames with multiple detections
        clean_data = track_data[~track_data['frame'].isin(multiple_detection_frames)]
        
        if len(clean_data) > 0:
            # Sort by frame
            clean_data = clean_data.sort_values('frame')
            
            # Interpolate gaps
            interpolated_data = interpolate_gaps(clean_data, max_gap)
            
            # Create output filename
            output_file = output_dir / f"vehicle_{track_id}.csv"
            
            # Save to CSV
            interpolated_data.to_csv(output_file, index=False)
            print(f"Saved track {track_id} to {output_file}")
            print(f"  - Original points: {len(track_data)}")
            print(f"  - After removing multiple detections: {len(clean_data)}")
            print(f"  - After interpolation: {len(interpolated_data)}")
        else:
            print(f"Skipping track {track_id} - no valid frames after removing multiple detections")

def main():
    parser = argparse.ArgumentParser(description='Process vehicle track data and create separate CSV files for each track')
    parser.add_argument('csv_path', help='Path to the input CSV file containing track data')
    parser.add_argument('--max-gap', type=int, default=3,
                      help='Maximum number of frames to interpolate between')
    args = parser.parse_args()
    
    process_track_data(args.csv_path, args.max_gap)

if __name__ == '__main__':
    main() 