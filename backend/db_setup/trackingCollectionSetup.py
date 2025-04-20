from flask import Flask, jsonify, request
from pymongo import MongoClient
import datetime
import uuid
import json
from bson import json_util

app = Flask(__name__)

# MongoDB connection string
MONGO_URI = "mongodb+srv://Temp:hack@cluster0.lhazyyp.mongodb.net/DriverData?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["DriverData"]
reports_collection = db["reports"]
movement_tracking_collection = db["movement_tracking"]

def parse_json(data):
    """Convert MongoDB results to JSON serializable format."""
    return json.loads(json_util.dumps(data))

@app.route("/")
def check_connection():
    """Check if MongoDB connection is successful."""
    try:
        # Verify connection by accessing server info
        server_info = client.server_info()
        
        return jsonify({
            "status": "success",
            "message": "Connected to MongoDB successfully",
            "server_version": server_info.get('version', 'Unknown'),
            "databases": client.list_database_names(),
            "collections": db.list_collection_names()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to connect to MongoDB: {str(e)}"
        }), 500

@app.route("/setup-indexes")
def setup_indexes():
    """Setup indexes for the movement tracking collection."""
    try:
        movement_tracking_collection.create_index("tracking_id", unique=True)
        movement_tracking_collection.create_index("report_id", unique=True)
        
        return jsonify({
            "status": "success",
            "message": "Indexes created successfully"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to create indexes: {str(e)}"
        }), 500

@app.route("/create-tracking", methods=["POST"])
def create_tracking():
    """Create a new movement tracking entry."""
    try:
        data = request.json
        
        # Validate required fields
        if 'report_id' not in data:
            return jsonify({
                "status": "error",
                "message": "report_id is required"
            }), 400
            
        report_id = data['report_id']
        
        # Check if report exists
        report = reports_collection.find_one({"report_id": report_id})
        if not report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Check if tracking for this report already exists
        existing_tracking = movement_tracking_collection.find_one({"report_id": report_id})
        if existing_tracking:
            return jsonify({
                "status": "error",
                "message": f"Tracking already exists for report {report_id}"
            }), 409
            
        # Extract relevant data from report
        confirm_bool = report.get("confirm_bool", "null")
        tracked_movements = report.get("tracked_movements", {})
        
        # Create tracking document
        tracking_id = str(uuid.uuid4())
        tracking = {
            "tracking_id": tracking_id,
            "report_id": report_id,
            "confirm_bool": confirm_bool,
            "movement_tracking": tracked_movements,
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        # Insert tracking into MongoDB
        result = movement_tracking_collection.insert_one(tracking)
        
        if result.acknowledged:
            return jsonify({
                "status": "success",
                "message": "Tracking created successfully",
                "tracking_id": tracking_id
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to create tracking"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error creating tracking: {str(e)}"
        }), 500

@app.route("/get-tracking/<tracking_id>", methods=["GET"])
def get_tracking(tracking_id):
    """Get a specific tracking entry by ID."""
    try:
        tracking = movement_tracking_collection.find_one({"tracking_id": tracking_id}, {'_id': 0})
        
        if not tracking:
            return jsonify({
                "status": "error",
                "message": f"Tracking with ID {tracking_id} not found"
            }), 404
            
        return jsonify({
            "status": "success",
            "tracking": tracking
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving tracking: {str(e)}"
        }), 500

@app.route("/get-tracking-by-report/<report_id>", methods=["GET"])
def get_tracking_by_report(report_id):
    """Get tracking entry by report ID."""
    try:
        tracking = movement_tracking_collection.find_one({"report_id": report_id}, {'_id': 0})
        
        if not tracking:
            return jsonify({
                "status": "error",
                "message": f"No tracking found for report ID {report_id}"
            }), 404
            
        return jsonify({
            "status": "success",
            "tracking": tracking
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving tracking: {str(e)}"
        }), 500

@app.route("/update-tracking/<tracking_id>", methods=["PUT"])
def update_tracking(tracking_id):
    """Update a tracking entry."""
    try:
        data = request.json
        
        # Check if tracking exists
        tracking = movement_tracking_collection.find_one({"tracking_id": tracking_id})
        if not tracking:
            return jsonify({
                "status": "error",
                "message": f"Tracking with ID {tracking_id} not found"
            }), 404
            
        # Update fields
        update_data = {
            "updated_at": datetime.datetime.utcnow()
        }
        
        for field in ['confirm_bool', 'movement_tracking']:
            if field in data:
                update_data[field] = data[field]
                
        # Update tracking
        result = movement_tracking_collection.update_one(
            {"tracking_id": tracking_id},
            {"$set": update_data}
        )
        
        if result.modified_count == 1:
            return jsonify({
                "status": "success",
                "message": "Tracking updated successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to update tracking"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating tracking: {str(e)}"
        }), 500

@app.route("/update-from-report/<report_id>", methods=["PUT"])
def update_from_report(report_id):
    """Update tracking entry from report data."""
    try:
        # Check if report exists
        report = reports_collection.find_one({"report_id": report_id})
        if not report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Extract relevant data from report
        confirm_bool = report.get("confirm_bool", "null")
        tracked_movements = report.get("tracked_movements", {})
        
        # Check if tracking exists
        tracking = movement_tracking_collection.find_one({"report_id": report_id})
        
        if not tracking:
            # Create new tracking if it doesn't exist
            tracking_id = str(uuid.uuid4())
            new_tracking = {
                "tracking_id": tracking_id,
                "report_id": report_id,
                "confirm_bool": confirm_bool,
                "movement_tracking": tracked_movements,
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
            
            result = movement_tracking_collection.insert_one(new_tracking)
            
            if result.acknowledged:
                return jsonify({
                    "status": "success",
                    "message": "New tracking created successfully",
                    "tracking_id": tracking_id
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to create tracking"
                }), 500
        
        # Update existing tracking
        update_data = {
            "confirm_bool": confirm_bool,
            "movement_tracking": tracked_movements,
            "updated_at": datetime.datetime.utcnow()
        }
        
        result = movement_tracking_collection.update_one(
            {"report_id": report_id},
            {"$set": update_data}
        )
        
        if result.modified_count == 1:
            return jsonify({
                "status": "success",
                "message": "Tracking updated successfully"
            })
        else:
            return jsonify({
                "status": "success",
                "message": "No changes made to tracking"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating tracking: {str(e)}"
        }), 500

@app.route("/delete-tracking/<tracking_id>", methods=["DELETE"])
def delete_tracking(tracking_id):
    """Delete a tracking entry."""
    try:
        # Check if tracking exists
        tracking = movement_tracking_collection.find_one({"tracking_id": tracking_id})
        if not tracking:
            return jsonify({
                "status": "error",
                "message": f"Tracking with ID {tracking_id} not found"
            }), 404
            
        # Delete tracking
        result = movement_tracking_collection.delete_one({"tracking_id": tracking_id})
        
        if result.deleted_count == 1:
            return jsonify({
                "status": "success",
                "message": "Tracking deleted successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to delete tracking"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error deleting tracking: {str(e)}"
        }), 500

@app.route("/create-all-from-reports", methods=["POST"])
def create_all_from_reports():
    """Create tracking entries for all reports that have movement data."""
    try:
        # Get all reports with tracked_movements data
        reports = reports_collection.find(
            {"tracked_movements": {"$exists": True, "$ne": {}}},
            {"report_id": 1, "confirm_bool": 1, "tracked_movements": 1}
        )
        
        created_count = 0
        skipped_count = 0
        
        for report in reports:
            report_id = report["report_id"]
            
            # Check if tracking already exists
            existing_tracking = movement_tracking_collection.find_one({"report_id": report_id})
            if existing_tracking:
                skipped_count += 1
                continue
                
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
        
        return jsonify({
            "status": "success",
            "message": f"Created {created_count} tracking entries, skipped {skipped_count} existing entries"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error creating tracking entries: {str(e)}"
        }), 500

@app.route("/clear-all", methods=["DELETE"])
def clear_all():
    """Delete all tracking entries."""
    try:
        result = movement_tracking_collection.delete_many({})
        
        return jsonify({
            "status": "success",
            "message": f"Deleted {result.deleted_count} tracking entries"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error clearing tracking entries: {str(e)}"
        }), 500

if __name__ == "__main__":
    # Create indexes
    movement_tracking_collection.create_index("tracking_id", unique=True)
    movement_tracking_collection.create_index("report_id", unique=True)
    
    app.run(debug=True, port=5003) 