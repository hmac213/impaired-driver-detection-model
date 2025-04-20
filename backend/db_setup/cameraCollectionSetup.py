from flask import Flask, jsonify, request
from pymongo import MongoClient
import pandas as pd
from pathlib import Path
import json

app = Flask(__name__)

# MongoDB connection string
MONGO_URI = "mongodb+srv://Temp:hack@cluster0.lhazyyp.mongodb.net/DriverData?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["DriverData"]
camera_collection = db["camera_db"]

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

@app.route("/upload-track-data", methods=["POST"])
def upload_track_data():
    """Upload track data from a CSV file with camera_id."""
    try:
        # Get camera_id from request
        camera_id = request.form.get('camera_id')
        
        if not camera_id:
            return jsonify({
                "status": "error",
                "message": "camera_id is required"
            }), 400
            
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
            
            # Add camera_id to each row
            df['camera_id'] = camera_id
            
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            # Insert data into MongoDB
            result = camera_collection.insert_many(records)
            
            return jsonify({
                "status": "success",
                "message": f"Successfully uploaded {len(result.inserted_ids)} records",
                "camera_id": camera_id
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Only CSV files are allowed"
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error uploading data: {str(e)}"
        }), 500

@app.route("/upload-json", methods=["POST"])
def upload_json_data():
    """Upload track data directly as JSON."""
    try:
        # Get data from request
        data = request.json
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400
            
        # Ensure data is in list format
        if not isinstance(data, list):
            data = [data]
            
        # Ensure each record has camera_id
        for record in data:
            if 'camera_id' not in record:
                return jsonify({
                    "status": "error",
                    "message": "camera_id is required for each record"
                }), 400
        
        # Insert data into MongoDB
        result = camera_collection.insert_many(data)
        
        return jsonify({
            "status": "success",
            "message": f"Successfully uploaded {len(result.inserted_ids)} records"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error uploading data: {str(e)}"
        }), 500

@app.route("/convert-csv", methods=["POST"])
def convert_csv_to_db():
    """Convert an existing CSV file to database entries with camera_id."""
    try:
        data = request.json
        
        if 'file_path' not in data or 'camera_id' not in data:
            return jsonify({
                "status": "error",
                "message": "file_path and camera_id are required"
            }), 400
            
        file_path = data['file_path']
        camera_id = data['camera_id']
        
        # Check if file exists
        path = Path(file_path)
        if not path.exists():
            return jsonify({
                "status": "error",
                "message": f"File not found: {file_path}"
            }), 404
            
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Add camera_id to each row
        df['camera_id'] = camera_id
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Insert data into MongoDB
        result = camera_collection.insert_many(records)
        
        return jsonify({
            "status": "success",
            "message": f"Successfully uploaded {len(result.inserted_ids)} records from {file_path}",
            "camera_id": camera_id
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error converting CSV: {str(e)}"
        }), 500

@app.route("/get-camera-data/<camera_id>")
def get_camera_data(camera_id):
    """Retrieve data for a specific camera."""
    try:
        # Query MongoDB for data
        data = list(camera_collection.find({"camera_id": camera_id}, {"_id": 0}))
        
        return jsonify({
            "status": "success",
            "camera_id": camera_id,
            "count": len(data),
            "data": data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving data: {str(e)}"
        }), 500

@app.route("/get-tracks/<track_id>")
def get_track_data(track_id):
    """Retrieve data for a specific track."""
    try:
        # Convert track_id to int if it's numeric
        if track_id.isdigit():
            track_id = int(track_id)
            
        # Query MongoDB for data
        data = list(camera_collection.find({"track_id": track_id}, {"_id": 0}))
        
        return jsonify({
            "status": "success",
            "track_id": track_id,
            "count": len(data),
            "data": data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving data: {str(e)}"
        }), 500

@app.route("/stats")
def get_stats():
    """Get overall statistics from the database."""
    try:
        # Get count of documents
        total_count = camera_collection.count_documents({})
        
        # Get unique camera IDs
        cameras = camera_collection.distinct("camera_id")
        
        # Get distribution of vehicle classes
        pipeline = [
            {"$group": {"_id": "$class", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        class_distribution = list(camera_collection.aggregate(pipeline))
        
        # Get track count per camera
        pipeline = [
            {"$group": {"_id": {"camera": "$camera_id", "track": "$track_id"}}},
            {"$group": {"_id": "$_id.camera", "track_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        tracks_per_camera = list(camera_collection.aggregate(pipeline))
        
        return jsonify({
            "status": "success",
            "total_records": total_count,
            "cameras": cameras,
            "class_distribution": class_distribution,
            "tracks_per_camera": tracks_per_camera
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving stats: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True) 