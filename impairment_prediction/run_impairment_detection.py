import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict
import json
import torch

# Add the impairment_detection folder to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'impairment_detection'))

# Import the model and prediction function
from impairment_prediction.predict_impairment import predict_impairment, GRUModel

# Define paths
MODEL_PATH = Path(__file__).parent.parent / "impairment_detection_model" / "impaired_detection_model_v3.pth"
CSV_DIR = Path(__file__).parent.parent / "object_detection_pipeline" / "output" / "vehicle_tracks"
OUTPUT_DIR = CSV_DIR.parent

def process_track_file(track_file: Path, model) -> Dict:
    """
    Process a single vehicle track file and run impairment detection.
    
    Args:
        track_file: Path to the vehicle track CSV file
        model: Loaded impairment detection model
        
    Returns:
        Dictionary containing track ID and impairment prediction results
    """
    # Read the track data
    df = pd.read_csv(track_file)
    
    # Extract track ID from filename
    track_id = int(track_file.stem.split('_')[1])
    
    # Skip tracks with too few points
    if len(df) < 10:  # Minimum number of points required
        print(f"Track {track_id} skipped: Too few points ({len(df)})")
        return None
    
    # Rename time column to match expected format
    df = df.rename(columns={'time': 'time_from_start'})
    
    # Ensure time_from_start is in seconds and starts from 0
    df['time_from_start'] = df['time_from_start'] - df['time_from_start'].min()
    
    # Skip tracks that are too short in duration
    duration = df['time_from_start'].max() - df['time_from_start'].min()
    if duration < 1.0:  # Minimum duration of 1 second
        print(f"Track {track_id} skipped: Too short duration ({duration:.2f} seconds)")
        return None
    
    # Normalize x coordinates to be between 0 and 1
    df['x'] = (df['x'] - df['x'].min()) / (df['x'].max() - df['x'].min())
    
    # Normalize y coordinates (assuming road width of 6 meters)
    df['y'] = df['y'] / 6.0
    
    # Run impairment detection
    try:
        probability = predict_impairment(model, df, device='cpu')
    except Exception as e:
        print(f"Error predicting for track {track_id}: {str(e)}")
        return None
    
    return {
        'track_id': track_id,
        'prediction': probability,
        'num_points': len(df),
        'duration': duration
    }

def run_impairment_detection():
    """
    Main function to run impairment detection on all vehicle track files.
    """
    # Check if paths exist
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found: {MODEL_PATH}")
        return
        
    if not CSV_DIR.exists():
        print(f"Error: CSV directory not found: {CSV_DIR}")
        return
        
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Initialize model
        model = GRUModel(2, hidden_size=64, num_layers=2, dropout=0.3)
        # Load state dict
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Get all vehicle track CSV files
    track_files = list(CSV_DIR.glob("vehicle_*.csv"))
    
    if not track_files:
        print(f"No vehicle track files found in {CSV_DIR}")
        return
    
    print(f"Found {len(track_files)} vehicle track files")
    
    # Process each track file
    results = []
    for track_file in track_files:
        print(f"Processing {track_file.name}...")
        try:
            result = process_track_file(track_file, model)
            if result is not None:  # Only append if processing was successful
                results.append(result)
                print(f"Track {result['track_id']}: {result['prediction']:.4f}")
        except Exception as e:
            print(f"Error processing {track_file.name}: {str(e)}")
    
    if not results:
        print("No valid tracks found to process")
        return
    
    # Save results to JSON file
    output_file = OUTPUT_DIR / "impairment_predictions.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\nSummary:")
    for result in results:
        print(f"Track {result['track_id']}:")
        print(f"  Probability: {result['prediction']:.4f}")
        print(f"  Classification: {'Impaired' if result['prediction'] > 0.5 else 'Not Impaired'}")
        print(f"  Duration: {result['duration']:.2f} seconds")
        print(f"  Points: {result['num_points']}")

if __name__ == '__main__':
    run_impairment_detection() 