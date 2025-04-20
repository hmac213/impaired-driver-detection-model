import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob
import os

# Get all CSV files in the vehicle tracks directory
track_files = glob.glob('object_detection_pipeline/output/vehicle_tracks/*.csv')
print(f"Found {len(track_files)} track files:")
for f in track_files:
    print(f"  - {f}")

if not track_files:
    print("No track files found! Check if the path is correct.")
    exit(1)

# Create a figure with equal aspect ratio
plt.figure(figsize=(15, 8))
ax = plt.gca()
ax.set_aspect('equal')  # This ensures one-to-one unit scale

# Plot each vehicle's trajectory
for track_file in tqdm(track_files, desc="Plotting trajectories"):
    # Read the track data
    vehicle_data = pd.read_csv(track_file)
    print(f"\nPlotting {track_file}")
    print(f"Data shape: {vehicle_data.shape}")
    print(f"Columns: {vehicle_data.columns.tolist()}")
    print(f"First few rows:\n{vehicle_data.head()}")
    
    # Get vehicle ID from filename
    vehicle_id = os.path.basename(track_file).split('_')[1].split('.')[0]
    
    # Plot trajectory
    plt.plot(vehicle_data['x'] * 25, vehicle_data['y'], 
             label=f'Vehicle {vehicle_id}',
             alpha=0.7, 
             linewidth=2,
             marker='o',
             markersize=4)

# Add labels and title
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Vehicle Trajectories')

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.grid(True, alpha=0.3)
plt.tight_layout()  # Adjust layout to make room for legend
plt.show() 