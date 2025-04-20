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
        
        print("Connected to MongoDB successfully!")
        
        # Clear existing reports
        result = reports_collection.delete_many({})
        print(f"Cleared {result.deleted_count} reports")
        
        # Create sample reports
        sample_reports = [
            {
                "report_id": str(uuid.uuid4()),
                "latitude": 37.7749,
                "longitude": -122.4194,
                "probability": 0.92,
                "descriptor": "drunk_driving",
                "confirm_bool": "impaired",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            },
            {
                "report_id": str(uuid.uuid4()),
                "latitude": 37.7833,
                "longitude": -122.4167,
                "probability": 0.78,
                "descriptor": "texting",
                "confirm_bool": "null",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            },
            {
                "report_id": str(uuid.uuid4()),
                "latitude": 37.7850,
                "longitude": -122.4200,
                "probability": 0.85,
                "descriptor": "drowsy",
                "confirm_bool": "safe",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
        ]
        
        # Insert sample reports
        result = reports_collection.insert_many(sample_reports)
        print(f"Created {len(result.inserted_ids)} sample reports")
        
        # Retrieve and print reports
        reports = list(reports_collection.find({}, {'_id': 0}))
        print("\nReports in database:")
        for report in reports:
            print(json.dumps(parse_json(report), indent=2))
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 