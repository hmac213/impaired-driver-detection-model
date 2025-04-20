import datetime
import uuid
from backend.config.database import movement_tracking_collection, reports_collection

class MovementTrackingModel:
    """Model for handling movement tracking data with references to reports."""
    
    @staticmethod
    def create_tracking(report_id):
        """Create a new movement tracking entry referencing a report."""
        # Check if report exists
        report = reports_collection.find_one({"report_id": report_id})
        if not report:
            return {"success": False, "error": f"Report with ID {report_id} not found"}
        
        # Check if tracking for this report already exists
        existing_tracking = movement_tracking_collection.find_one({"report_id": report_id})
        if existing_tracking:
            return {"success": False, "error": f"Tracking already exists for report {report_id}"}
        
        # Extract relevant data from report
        confirm_bool = report.get("confirm_bool", "null")
        tracked_movements = report.get("tracked_movements", {})
        
        # Create tracking document
        tracking = {
            "tracking_id": str(uuid.uuid4()),
            "report_id": report_id,
            "confirm_bool": confirm_bool,
            "movement_tracking": tracked_movements,
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        # Insert tracking into MongoDB
        result = movement_tracking_collection.insert_one(tracking)
        
        if result.acknowledged:
            return {"success": True, "tracking_id": tracking["tracking_id"]}
        else:
            return {"success": False, "error": "Failed to create tracking"}
    
    @staticmethod
    def update_tracking_from_report(report_id):
        """Update tracking data when a report is updated."""
        # Check if report exists
        report = reports_collection.find_one({"report_id": report_id})
        if not report:
            return {"success": False, "error": f"Report with ID {report_id} not found"}
        
        # Check if tracking for this report exists
        existing_tracking = movement_tracking_collection.find_one({"report_id": report_id})
        if not existing_tracking:
            # Create new tracking if it doesn't exist
            return MovementTrackingModel.create_tracking(report_id)
        
        # Extract relevant data from report
        confirm_bool = report.get("confirm_bool", "null")
        tracked_movements = report.get("tracked_movements", {})
        
        # Update tracking document
        update_data = {
            "confirm_bool": confirm_bool,
            "movement_tracking": tracked_movements,
            "updated_at": datetime.datetime.utcnow()
        }
        
        # Update tracking in MongoDB
        result = movement_tracking_collection.update_one(
            {"report_id": report_id},
            {"$set": update_data}
        )
        
        return {
            "success": result.modified_count == 1,
            "matched": result.matched_count > 0
        }
    
    @staticmethod
    def get_all_tracking():
        """Get all movement tracking entries."""
        tracking_entries = list(movement_tracking_collection.find({}, {'_id': 0}))
        return tracking_entries
    
    @staticmethod
    def get_tracking_by_id(tracking_id):
        """Get a specific tracking entry by ID."""
        tracking = movement_tracking_collection.find_one({"tracking_id": tracking_id}, {'_id': 0})
        return tracking
    
    @staticmethod
    def get_tracking_by_report_id(report_id):
        """Get tracking entry by report ID."""
        tracking = movement_tracking_collection.find_one({"report_id": report_id}, {'_id': 0})
        return tracking
    
    @staticmethod
    def delete_tracking(tracking_id):
        """Delete a specific tracking entry."""
        result = movement_tracking_collection.delete_one({"tracking_id": tracking_id})
        
        return {
            "success": result.deleted_count == 1,
            "deleted": result.deleted_count > 0
        }
    
    @staticmethod
    def delete_tracking_by_report_id(report_id):
        """Delete tracking entry by report ID."""
        result = movement_tracking_collection.delete_one({"report_id": report_id})
        
        return {
            "success": result.deleted_count == 1,
            "deleted": result.deleted_count > 0
        }
        
    @staticmethod
    def create_all_from_reports():
        """Create tracking entries for all reports that have movement data."""
        # Get all reports with tracked_movements data
        reports = reports_collection.find(
            {"tracked_movements": {"$exists": True, "$ne": {}}},
            {"report_id": 1, "confirm_bool": 1, "tracked_movements": 1}
        )
        
        created_count = 0
        skipped_count = 0
        
        for report in reports:
            report_id = report["report_id"]
            
            # Check if tracking for this report already exists
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
        
        return {
            "success": True,
            "created_count": created_count,
            "skipped_count": skipped_count
        }
    
    @staticmethod
    def update_all_from_reports():
        """Update all tracking entries from their corresponding reports."""
        # Get all tracking entries
        tracking_entries = list(movement_tracking_collection.find({}, {"report_id": 1}))
        
        updated_count = 0
        not_found_count = 0
        
        for tracking in tracking_entries:
            report_id = tracking["report_id"]
            
            # Get corresponding report
            report = reports_collection.find_one({"report_id": report_id})
            
            if not report:
                not_found_count += 1
                continue
                
            # Update tracking document
            update_data = {
                "confirm_bool": report.get("confirm_bool", "null"),
                "movement_tracking": report.get("tracked_movements", {}),
                "updated_at": datetime.datetime.utcnow()
            }
            
            # Update tracking in MongoDB
            result = movement_tracking_collection.update_one(
                {"report_id": report_id},
                {"$set": update_data}
            )
            
            if result.modified_count == 1:
                updated_count += 1
        
        return {
            "success": True,
            "updated_count": updated_count,
            "not_found_count": not_found_count
        }
    
    @staticmethod
    def clear_all_tracking():
        """Delete all tracking entries."""
        result = movement_tracking_collection.delete_many({})
        
        return {
            "success": True,
            "deleted_count": result.deleted_count
        } 