from flask import Flask, jsonify, request
from pymongo import MongoClient
import datetime
import uuid
import pandas as pd
import json
from pathlib import Path

app = Flask(__name__)

# MongoDB connection string
MONGO_URI = "mongodb+srv://Temp:hack@cluster0.lhazyyp.mongodb.net/DriverData?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["DriverData"]
reports_collection = db["reports"]

# Valid status values for confirm_bool field
VALID_STATUS_VALUES = ["null", "safe", "impaired"]

def parse_csv_to_json(file_path):
    """Parse CSV file to JSON format for tracked_movements."""
    try:
        df = pd.read_csv(file_path)
        # Group by track_id to create nested structure
        tracks = {}
        for track_id, group in df.groupby('track_id'):
            tracks[str(track_id)] = group.to_dict('records')
        return tracks
    except Exception as e:
        print(f"Error parsing CSV: {str(e)}")
        return {}

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

@app.route("/create-report", methods=["POST"])
def create_report():
    """Create a new report in the database."""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['latitude', 'longitude', 'probability', 'descriptor']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "status": "error",
                    "message": f"Missing required field: {field}"
                }), 400
                
        # Generate a unique report_id
        report_id = str(uuid.uuid4())
        
        # Get confirm_bool value and validate
        confirm_bool = data.get('confirm_bool', "null")  # Default to "null" if not provided
        if confirm_bool not in VALID_STATUS_VALUES:
            return jsonify({
                "status": "error",
                "message": f"Invalid confirm_bool value. Must be one of: {', '.join(VALID_STATUS_VALUES)}"
            }), 400
        
        # Handle tracked_movements if provided
        tracked_movements = {}
        if 'tracked_movements' in data:
            tracked_movements = data['tracked_movements']
        elif 'csv_path' in data:
            # Parse CSV file if path is provided
            csv_path = data['csv_path']
            tracked_movements = parse_csv_to_json(csv_path)
        
        # Create report document
        report = {
            "report_id": report_id,
            "latitude": data['latitude'],
            "longitude": data['longitude'],
            "probability": data['probability'],
            "descriptor": data['descriptor'],
            "confirm_bool": confirm_bool,
            "tracked_movements": tracked_movements,
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        # Insert report into MongoDB
        result = reports_collection.insert_one(report)
        
        if result.acknowledged:
            return jsonify({
                "status": "success",
                "message": "Report created successfully",
                "report_id": report_id
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to create report"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error creating report: {str(e)}"
        }), 500

@app.route("/upload-track-data/<report_id>", methods=["POST"])
def upload_track_data(report_id):
    """Upload track data from a CSV file and attach to a report."""
    try:
        # Check if report exists
        existing_report = reports_collection.find_one({"report_id": report_id})
        
        if not existing_report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file part"
            }), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No selected file"
            }), 400
            
        if file and file.filename.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Group by track_id to create nested structure
            tracks = {}
            for track_id, group in df.groupby('track_id'):
                tracks[str(track_id)] = group.to_dict('records')
            
            # Update the report with tracked_movements
            result = reports_collection.update_one(
                {"report_id": report_id},
                {"$set": {
                    "tracked_movements": tracks,
                    "updated_at": datetime.datetime.utcnow()
                }}
            )
            
            if result.modified_count == 1:
                return jsonify({
                    "status": "success",
                    "message": f"Successfully uploaded track data for report {report_id}",
                    "tracks_count": len(tracks)
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to update report with track data"
                }), 500
        else:
            return jsonify({
                "status": "error",
                "message": "Only CSV files are allowed"
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error uploading track data: {str(e)}"
        }), 500

@app.route("/reports", methods=["GET"])
def get_reports():
    """Get all reports."""
    try:
        # Get query parameters
        status = request.args.get('status')
        min_probability = request.args.get('min_probability')
        descriptor = request.args.get('descriptor')
        include_tracks = request.args.get('include_tracks', 'false').lower() == 'true'
        
        # Build query
        query = {}
        
        if status is not None:
            if status in VALID_STATUS_VALUES:
                query['confirm_bool'] = status
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid status parameter. Must be one of: {', '.join(VALID_STATUS_VALUES)}"
                }), 400
            
        if min_probability is not None:
            query['probability'] = {'$gte': float(min_probability)}
            
        if descriptor is not None:
            query['descriptor'] = descriptor
        
        # Define projection (fields to include)
        projection = {'_id': 0}
        if not include_tracks:
            projection['tracked_movements'] = 0
            
        # Query MongoDB for reports
        reports = list(reports_collection.find(query, projection))
        
        return jsonify({
            "status": "success",
            "count": len(reports),
            "reports": reports
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving reports: {str(e)}"
        }), 500

@app.route("/reports/<report_id>", methods=["GET"])
def get_report(report_id):
    """Get a specific report by ID."""
    try:
        # Check if we should include track data
        include_tracks = request.args.get('include_tracks', 'true').lower() == 'true'
        
        # Define projection (fields to include)
        projection = {'_id': 0}
        if not include_tracks:
            projection['tracked_movements'] = 0
            
        # Query MongoDB for the report
        report = reports_collection.find_one({"report_id": report_id}, projection)
        
        if not report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        return jsonify({
            "status": "success",
            "report": report
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving report: {str(e)}"
        }), 500

@app.route("/reports/<report_id>", methods=["PUT"])
def update_report(report_id):
    """Update a specific report."""
    try:
        data = request.json
        
        # Query MongoDB for the report
        existing_report = reports_collection.find_one({"report_id": report_id})
        
        if not existing_report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Update fields
        update_data = {
            "updated_at": datetime.datetime.utcnow()
        }
        
        # Validate confirm_bool if provided
        if 'confirm_bool' in data:
            if data['confirm_bool'] not in VALID_STATUS_VALUES:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid confirm_bool value. Must be one of: {', '.join(VALID_STATUS_VALUES)}"
                }), 400
            update_data['confirm_bool'] = data['confirm_bool']
        
        # Handle tracked_movements if provided
        if 'tracked_movements' in data:
            update_data['tracked_movements'] = data['tracked_movements']
        elif 'csv_path' in data:
            # Parse CSV file if path is provided
            csv_path = data['csv_path']
            update_data['tracked_movements'] = parse_csv_to_json(csv_path)
        
        # Only update other provided fields
        for field in ['latitude', 'longitude', 'probability', 'descriptor']:
            if field in data:
                update_data[field] = data[field]
                
        # Update the report
        result = reports_collection.update_one(
            {"report_id": report_id},
            {"$set": update_data}
        )
        
        if result.modified_count == 1:
            return jsonify({
                "status": "success",
                "message": "Report updated successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to update report"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating report: {str(e)}"
        }), 500

@app.route("/reports/<report_id>", methods=["DELETE"])
def delete_report(report_id):
    """Delete a specific report."""
    try:
        # Query MongoDB for the report
        existing_report = reports_collection.find_one({"report_id": report_id})
        
        if not existing_report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Delete the report
        result = reports_collection.delete_one({"report_id": report_id})
        
        if result.deleted_count == 1:
            return jsonify({
                "status": "success",
                "message": "Report deleted successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to delete report"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error deleting report: {str(e)}"
        }), 500

@app.route("/reports/mark-as/<report_id>", methods=["PUT"])
def mark_report_status(report_id):
    """Mark a report with a specific status."""
    try:
        data = request.json
        if not data or 'status' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing status parameter"
            }), 400
            
        requested_status = data['status']
        if requested_status not in VALID_STATUS_VALUES:
            return jsonify({
                "status": "error",
                "message": f"Invalid status value. Must be one of: {', '.join(VALID_STATUS_VALUES)}"
            }), 400
            
        # Update the report
        result = reports_collection.update_one(
            {"report_id": report_id},
            {"$set": {
                "confirm_bool": requested_status,
                "updated_at": datetime.datetime.utcnow()
            }}
        )
        
        if result.matched_count == 0:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        if result.modified_count == 1:
            return jsonify({
                "status": "success",
                "message": f"Report marked as '{requested_status}' successfully"
            })
        else:
            return jsonify({
                "status": "info",
                "message": f"Report was already marked as '{requested_status}'"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error marking report: {str(e)}"
        }), 500

@app.route("/reports/mark-impaired/<report_id>", methods=["PUT"])
def mark_report_impaired(report_id):
    """Mark a report as impaired (legacy support for /confirm endpoint)."""
    try:
        # Update the report
        result = reports_collection.update_one(
            {"report_id": report_id},
            {"$set": {
                "confirm_bool": "impaired",
                "updated_at": datetime.datetime.utcnow()
            }}
        )
        
        if result.matched_count == 0:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        if result.modified_count == 1:
            return jsonify({
                "status": "success",
                "message": "Report marked as impaired successfully"
            })
        else:
            return jsonify({
                "status": "info",
                "message": "Report was already marked as impaired"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error marking report: {str(e)}"
        }), 500

@app.route("/reports/stats", methods=["GET"])
def get_report_stats():
    """Get statistics about reports."""
    try:
        # Get total count
        total_count = reports_collection.count_documents({})
        
        # Get counts by status
        null_count = reports_collection.count_documents({"confirm_bool": "null"})
        safe_count = reports_collection.count_documents({"confirm_bool": "safe"})
        impaired_count = reports_collection.count_documents({"confirm_bool": "impaired"})
        
        # Count reports with tracked_movements
        with_tracks_count = reports_collection.count_documents({"tracked_movements": {"$exists": True, "$ne": {}}})
        
        # Get distribution by descriptor
        pipeline = [
            {"$group": {"_id": "$descriptor", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        descriptor_distribution = list(reports_collection.aggregate(pipeline))
        
        # Get high probability reports (>= 0.8)
        high_prob_count = reports_collection.count_documents({"probability": {"$gte": 0.8}})
        
        return jsonify({
            "status": "success",
            "total_reports": total_count,
            "status_distribution": {
                "null": null_count,
                "safe": safe_count,
                "impaired": impaired_count
            },
            "reports_with_tracks": with_tracks_count,
            "high_probability_reports": high_prob_count,
            "descriptor_distribution": descriptor_distribution
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving stats: {str(e)}"
        }), 500

# Create some sample reports for testing
@app.route("/create-sample-reports", methods=["POST"])
def create_sample_reports():
    """Create sample reports for testing."""
    try:
        # Check if a CSV file for tracks is specified
        csv_path = request.json.get('csv_path') if request.json else None
        tracked_movements = {}
        
        # Parse CSV file if provided
        if csv_path:
            tracked_movements = parse_csv_to_json(csv_path)
        
        sample_reports = [
            {
                "report_id": str(uuid.uuid4()),
                "latitude": 37.7749,
                "longitude": -122.4194,
                "probability": 0.92,
                "descriptor": "drunk_driving",
                "confirm_bool": "impaired",
                "tracked_movements": tracked_movements.copy() if tracked_movements else {},
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
                "tracked_movements": tracked_movements.copy() if tracked_movements else {},
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
                "tracked_movements": tracked_movements.copy() if tracked_movements else {},
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
        ]
        
        # Insert sample reports
        result = reports_collection.insert_many(sample_reports)
        
        return jsonify({
            "status": "success",
            "message": f"Created {len(result.inserted_ids)} sample reports",
            "with_tracks": bool(tracked_movements)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error creating sample reports: {str(e)}"
        }), 500

@app.route("/update-legacy-reports", methods=["POST"])
def update_legacy_reports():
    """Update existing reports with boolean confirm_bool values to use strings."""
    try:
        # Find all reports with boolean confirm_bool
        boolean_reports = list(reports_collection.find({
            "$or": [
                {"confirm_bool": True},
                {"confirm_bool": False}
            ]
        }))
        
        updated_count = 0
        
        for report in boolean_reports:
            # Convert boolean to string
            status = "impaired" if report["confirm_bool"] is True else "null"
            
            # Update the report
            result = reports_collection.update_one(
                {"_id": report["_id"]},
                {"$set": {
                    "confirm_bool": status,
                    "updated_at": datetime.datetime.utcnow()
                }}
            )
            
            if result.modified_count == 1:
                updated_count += 1
                
        return jsonify({
            "status": "success",
            "message": f"Updated {updated_count} reports to use new status values",
            "updated_count": updated_count
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating reports: {str(e)}"
        }), 500

@app.route("/reports/tracks/<report_id>", methods=["GET"])
def get_report_tracks(report_id):
    """Get just the tracked movements for a specific report."""
    try:
        # Query MongoDB for the report
        report = reports_collection.find_one({"report_id": report_id}, {"_id": 0, "tracked_movements": 1})
        
        if not report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Check if tracked_movements field exists
        tracked_movements = report.get("tracked_movements", {})
        
        return jsonify({
            "status": "success",
            "report_id": report_id,
            "tracked_movements": tracked_movements
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving track data: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001) 