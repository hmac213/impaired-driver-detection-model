from pymongo import MongoClient
import datetime
import uuid
import json

# MongoDB connection string
MONGO_URI = "mongodb+srv://Temp:hack@cluster0.lhazyyp.mongodb.net/DriverData?retryWrites=true&w=majority&appName=Cluster0"

def main():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client["DriverData"]
        camera_collection = db["camera_db"]
        calibration_collection = db["camera_calibration"]
        
        print("Connected to MongoDB successfully!")
        
        # Get all unique camera IDs from camera_collection
        camera_ids = camera_collection.distinct("camera_id")
        print(f"Found {len(camera_ids)} unique cameras in the database")
        
        if len(camera_ids) == 0:
            print("No cameras found in the database. Please add camera data first.")
            return
            
        # Process each camera
        for camera_id in camera_ids:
            print(f"\nProcessing camera: {camera_id}")
            
            # Check if camera already has a calibration
            existing_calibration = calibration_collection.find_one({"camera_id": camera_id})
            if existing_calibration:
                print(f"  Camera {camera_id} already has calibration data with ID: {existing_calibration['calibration_id']}")
                continue
                
            # Create sample calibration data for this camera
            calibration_id = str(uuid.uuid4())
            timestamp = datetime.datetime.utcnow()
            
            # Sample lane polynomials (coefficients for quadratic functions)
            # These represent 2 lane lines as polynomial functions
            lane_polynomials = [
                [0.0002, -0.3, 100],  # x = 0.0002*y² - 0.3*y + 100
                [0.0002, -0.3, 300]   # x = 0.0002*y² - 0.3*y + 300
            ]
            
            # Sample lane points that generated the polynomials
            lane_points = [
                [[100, 200], [90, 300], [70, 400]],  # Points for first lane
                [[300, 200], [290, 300], [270, 400]] # Points for second lane
            ]
            
            # Rectangle points for region of interest (in image coordinates)
            rectangle_points = [
                [50, 200],    # Top-left
                [350, 200],   # Top-right
                [400, 400],   # Bottom-right
                [0, 400]      # Bottom-left
            ]
            
            # Homography matrix (3x3) for transforming image coordinates to world coordinates
            homography_matrix = [
                [0.015, 0.002, -1.5],
                [0.001, 0.025, -5.0],
                [0.000, 0.001, 1.0]
            ]
            
            # Inverse homography matrix
            inverse_homography_matrix = [
                [67.8, -5.4, 104.2],
                [-2.7, 40.5, 202.5],
                [0.0, 0.0, 1.0]
            ]
            
            # Create the calibration document
            calibration_data = {
                "calibration_id": calibration_id,
                "camera_id": camera_id,
                "location_name": f"Location for camera {camera_id}",
                "timestamp": timestamp,
                "created_at": timestamp,
                
                # Lane data
                "lane_data": {
                    "lane_polynomials": lane_polynomials,
                    "lane_points": lane_points
                },
                
                # ROI data
                "roi_data": {
                    "rectangle_points": rectangle_points,
                    "entrance_side": 0,  # Index of the entrance side (0=top)
                    "exit_side": 2,      # Index of the exit side (2=bottom)
                    "world_width": 3.7,  # Standard lane width in meters
                    "world_length": 30.0 # Length of ROI in meters
                },
                
                # Homography data
                "homography_data": {
                    "matrix": homography_matrix,
                    "inverse_matrix": inverse_homography_matrix
                },
                
                # Reference point for vehicle detection
                "reference_point": {
                    "rel_x": 0.5,  # Center of the bounding box horizontally
                    "rel_y": 1.0   # Bottom of the bounding box vertically
                },
                
                # Set this calibration as active
                "is_active": True
            }
            
            # Insert the calibration document
            result = calibration_collection.insert_one(calibration_data)
            
            if result.acknowledged:
                print(f"  Created calibration {calibration_id} for camera {camera_id}")
                
                # Update camera entries to reference this calibration
                update_result = camera_collection.update_many(
                    {"camera_id": camera_id},
                    {"$set": {"calibration_id": calibration_id}}
                )
                
                print(f"  Updated {update_result.modified_count} camera entries with calibration reference")
            else:
                print(f"  Failed to create calibration for camera {camera_id}")
        
        print("\nCalibration update complete!")
        
        # Verify updates
        print("\nVerifying updates:")
        for camera_id in camera_ids:
            # Get first camera entry
            camera_entry = camera_collection.find_one({"camera_id": camera_id}, {"_id": 0, "camera_id": 1, "calibration_id": 1})
            if camera_entry and "calibration_id" in camera_entry:
                print(f"  Camera {camera_id} now references calibration: {camera_entry['calibration_id']}")
            else:
                print(f"  Camera {camera_id} does not have a calibration reference")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 