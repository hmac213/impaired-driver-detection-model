from pymongo import MongoClient
import pandas as pd
import datetime
import json
from bson import json_util

# MongoDB connection string
MONGO_URI = "mongodb+srv://Temp:hack@cluster0.lhazyyp.mongodb.net/DriverData?retryWrites=true&w=majority&appName=Cluster0"

# Path to the tracks CSV file
TRACKS_CSV_PATH = "../object_detection_pipeline/output/test_tracks.csv"

def parse_json(data):
    """Convert MongoDB results to JSON serializable format."""
    return json.loads(json_util.dumps(data))

def main():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client["DriverData"]
        reports_collection = db["reports"]
        
        print("Connected to MongoDB successfully!")
        
        # Read the CSV file
        df = pd.read_csv(TRACKS_CSV_PATH)
        print(f"Loaded tracks CSV with {len(df)} rows")
        
        # Group by track_id
        tracks = {}
        for track_id in [1, 2, 3]:  # We only need tracks 1, 2, and 3
            track_data = df[df['track_id'] == track_id]
            if not track_data.empty:
                tracks[str(track_id)] = track_data.to_dict('records')
                print(f"Track {track_id}: {len(track_data)} points")
            else:
                print(f"No data found for track {track_id}")
        
        # Get the first 3 reports from the database
        reports = list(reports_collection.find().limit(3))
        
        if len(reports) < 3:
            print(f"Warning: Only found {len(reports)} reports, need 3 reports")
            if len(reports) == 0:
                print("No reports found. Please create sample reports first.")
                return
        
        # Update each report with the corresponding track data
        for i, report in enumerate(reports[:3]):
            track_id = str(i + 1)  # Track 1, 2, or 3
            
            if track_id in tracks:
                print(f"Updating report {i+1} (ID: {report.get('report_id')}) with track {track_id}")
                
                # Update the report with the track data
                result = reports_collection.update_one(
                    {"_id": report["_id"]},
                    {"$set": {
                        "tracked_movements": {track_id: tracks[track_id]},
                        "updated_at": datetime.datetime.utcnow()
                    }}
                )
                
                if result.modified_count == 1:
                    print(f"Successfully updated report {i+1} with track {track_id}")
                else:
                    print(f"Failed to update report {i+1}")
            else:
                print(f"No track data available for track {track_id}")
        
        # Verify the updates
        updated_reports = list(reports_collection.find(
            {"tracked_movements": {"$exists": True, "$ne": {}}},
            {"_id": 0, "report_id": 1, "descriptor": 1, "confirm_bool": 1}
        ))
        
        print("\nUpdated reports:")
        for report in updated_reports:
            print(json.dumps(parse_json(report), indent=2))
        
        print(f"\nTotal reports with tracks: {len(updated_reports)}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 