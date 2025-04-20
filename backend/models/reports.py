import datetime
import uuid
from backend.config.database import reports_collection

class ReportsModel:
    """Model for handling reports collection operations."""
    
    @staticmethod
    def create_report(data):
        """Create a new report in the database."""
        # Generate a unique report_id
        report_id = str(uuid.uuid4())
        
        # Create report document
        report = {
            "report_id": report_id,
            "latitude": data['latitude'],
            "longitude": data['longitude'],
            "probability": data['probability'],
            "descriptor": data['descriptor'],
            "confirm_bool": data.get('confirm_bool', "null"),  # Default to "null" if not provided
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        # Validate status field
        status = report["confirm_bool"]
        if status not in ["null", "safe", "impaired"]:
            report["confirm_bool"] = "null"  # Default to "null" if invalid value
        
        # Insert report into MongoDB
        result = reports_collection.insert_one(report)
        
        if result.acknowledged:
            return {"success": True, "report_id": report_id}
        else:
            return {"success": False, "error": "Failed to create report"}
    
    @staticmethod
    def get_reports(query=None):
        """Get reports based on query parameters."""
        if query is None:
            query = {}
            
        # Query MongoDB for reports
        reports = list(reports_collection.find(query, {'_id': 0}))
        return reports
    
    @staticmethod
    def get_report_by_id(report_id):
        """Get a specific report by ID."""
        report = reports_collection.find_one({"report_id": report_id}, {'_id': 0})
        return report
    
    @staticmethod
    def update_report(report_id, update_data):
        """Update a specific report."""
        # Add updated timestamp
        update_data["updated_at"] = datetime.datetime.utcnow()
        
        # Validate status field if provided
        if "confirm_bool" in update_data:
            status = update_data["confirm_bool"]
            if status not in ["null", "safe", "impaired"]:
                update_data["confirm_bool"] = "null"
        
        # Update the report
        result = reports_collection.update_one(
            {"report_id": report_id},
            {"$set": update_data}
        )
        
        return {
            "success": result.modified_count == 1,
            "matched": result.matched_count > 0
        }
    
    @staticmethod
    def delete_report(report_id):
        """Delete a specific report."""
        result = reports_collection.delete_one({"report_id": report_id})
        
        return {
            "success": result.deleted_count == 1,
            "deleted": result.deleted_count > 0
        }
    
    @staticmethod
    def set_report_status(report_id, status):
        """Set status of a report (null, safe, or impaired)."""
        # Validate status
        if status not in ["null", "safe", "impaired"]:
            return {
                "success": False,
                "error": "Invalid status value. Must be 'null', 'safe', or 'impaired'."
            }
            
        result = reports_collection.update_one(
            {"report_id": report_id},
            {"$set": {
                "confirm_bool": status,
                "updated_at": datetime.datetime.utcnow()
            }}
        )
        
        return {
            "success": result.modified_count == 1,
            "matched": result.matched_count > 0
        }
    
    @staticmethod
    def mark_as_safe(report_id):
        """Mark a report as safe."""
        return ReportsModel.set_report_status(report_id, "safe")
    
    @staticmethod
    def mark_as_impaired(report_id):
        """Mark a report as impaired."""
        return ReportsModel.set_report_status(report_id, "impaired")
        
    @staticmethod
    def confirm_report(report_id):
        """Confirm a specific report (legacy method)."""
        # For backward compatibility, confirm_report now marks as "impaired"
        return ReportsModel.mark_as_impaired(report_id)
    
    @staticmethod
    def get_stats():
        """Get statistics about reports."""
        # Get total count
        total_count = reports_collection.count_documents({})
        
        # Get counts by status
        null_count = reports_collection.count_documents({"confirm_bool": "null"})
        safe_count = reports_collection.count_documents({"confirm_bool": "safe"})
        impaired_count = reports_collection.count_documents({"confirm_bool": "impaired"})
        
        # Get distribution by descriptor
        pipeline = [
            {"$group": {"_id": "$descriptor", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        descriptor_distribution = list(reports_collection.aggregate(pipeline))
        
        # Get high probability reports (>= 0.8)
        high_prob_count = reports_collection.count_documents({"probability": {"$gte": 0.8}})
        
        return {
            "total_reports": total_count,
            "status_distribution": {
                "null": null_count,
                "safe": safe_count,
                "impaired": impaired_count
            },
            "high_probability_reports": high_prob_count,
            "descriptor_distribution": descriptor_distribution
        }
    
    @staticmethod
    def create_sample_reports():
        """Create sample reports for testing."""
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
        
        return {
            "success": len(result.inserted_ids) > 0,
            "count": len(result.inserted_ids)
        }
        
    @staticmethod
    def update_existing_reports():
        """Update existing reports to use the new status values."""
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
                
        return {
            "success": True,
            "updated_count": updated_count
        }
    
    @staticmethod
    def clear_all_reports():
        """Delete all reports from the collection."""
        result = reports_collection.delete_many({})
        return {
            "success": True,
            "deleted_count": result.deleted_count
        } 