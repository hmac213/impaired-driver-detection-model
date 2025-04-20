from pymongo import MongoClient
import datetime
import uuid
import json
from bson import json_util

# MongoDB connection string
MONGO_URI = "mongodb+srv://Temp:hack@cluster0.lhazyyp.mongodb.net/DriverData?retryWrites=true&w=majority&appName=Cluster0"

def parse_json(data):
    """Convert MongoDB results to JSON serializable format."""
    return json.loads(json_util.dumps(data))

def main():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client["DriverData"]
        reports_collection = db["reports"]
        movement_tracking_collection = db["movement_tracking"]
        
        print("Connected to MongoDB successfully!")
        
        # Create indexes
        movement_tracking_collection.create_index("tracking_id", unique=True)
        movement_tracking_collection.create_index("report_id", unique=True)
        print("Created indexes for tracking_id and report_id")
        
        # Clear existing tracking entries
        clear_result = movement_tracking_collection.delete_many({})
        print(f"Cleared {clear_result.deleted_count} existing tracking entries")
        
        # Get all reports with tracked_movements data
        reports = list(reports_collection.find(
            {"tracked_movements": {"$exists": True, "$ne": {}}},
            {"report_id": 1, "confirm_bool": 1, "tracked_movements": 1}
        ))
        
        print(f"Found {len(reports)} reports with tracking data")
        
        if len(reports) == 0:
            print("No reports with tracking data found. Please add tracking data to reports first.")
            return
        
        # Create tracking entries
        created_count = 0
        
        for report in reports:
            report_id = report["report_id"]
            
            # Create tracking document
            tracking = {
                "tracking_id": str(uuid.uuid4()),
                "report_id": report_id,
                "confirm_bool": report.get("confirm_bool", "null"),
                "movement_tracking": report.get("tracked_movements", {}),
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
            
            # Insert tracking into MongoDB
            result = movement_tracking_collection.insert_one(tracking)
            
            if result.acknowledged:
                created_count += 1
        
        print(f"Created {created_count} tracking entries")
        
        # Verify tracking entries were created
        tracking_entries = list(movement_tracking_collection.find({}, {'_id': 0}))
        
        print("\nTracking entries in database:")
        for tracking in tracking_entries[:2]:  # Just show the first two to keep output manageable
            print(json.dumps(parse_json(tracking), indent=2))
            
        if len(tracking_entries) > 2:
            print(f"... and {len(tracking_entries) - 2} more")
            
        print("\nMovement tracking collection initialized successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 