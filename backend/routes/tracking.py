from flask import Blueprint, jsonify, request
from backend.models.movement_tracking import MovementTrackingModel
from backend.models.reports import ReportsModel

tracking_bp = Blueprint('tracking', __name__)

@tracking_bp.route('/', methods=['GET'])
def get_all_tracking():
    """Get all movement tracking entries."""
    try:
        tracking_entries = MovementTrackingModel.get_all_tracking()
        
        return jsonify({
            "status": "success",
            "count": len(tracking_entries),
            "tracking": tracking_entries
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving tracking entries: {str(e)}"
        }), 500

@tracking_bp.route('/<tracking_id>', methods=['GET'])
def get_tracking(tracking_id):
    """Get a specific tracking entry by ID."""
    try:
        tracking = MovementTrackingModel.get_tracking_by_id(tracking_id)
        
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

@tracking_bp.route('/report/<report_id>', methods=['GET'])
def get_tracking_by_report(report_id):
    """Get tracking entry for a specific report."""
    try:
        tracking = MovementTrackingModel.get_tracking_by_report_id(report_id)
        
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

@tracking_bp.route('/create/<report_id>', methods=['POST'])
def create_tracking(report_id):
    """Create a new tracking entry for a report."""
    try:
        # Check if report exists
        report = ReportsModel.get_report_by_id(report_id)
        
        if not report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Create tracking
        result = MovementTrackingModel.create_tracking(report_id)
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": "Tracking created successfully",
                "tracking_id": result["tracking_id"]
            })
        else:
            return jsonify({
                "status": "error",
                "message": result.get('error', 'Failed to create tracking')
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error creating tracking: {str(e)}"
        }), 500

@tracking_bp.route('/update/<report_id>', methods=['PUT'])
def update_tracking(report_id):
    """Update tracking from report data."""
    try:
        # Update tracking
        result = MovementTrackingModel.update_tracking_from_report(report_id)
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": "Tracking updated successfully"
            })
        elif 'tracking_id' in result:
            # This means a new tracking was created
            return jsonify({
                "status": "success",
                "message": "New tracking created successfully",
                "tracking_id": result["tracking_id"]
            })
        else:
            return jsonify({
                "status": "error",
                "message": result.get('error', 'Failed to update tracking')
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating tracking: {str(e)}"
        }), 500

@tracking_bp.route('/<tracking_id>', methods=['DELETE'])
def delete_tracking(tracking_id):
    """Delete a tracking entry."""
    try:
        # Check if tracking exists
        tracking = MovementTrackingModel.get_tracking_by_id(tracking_id)
        
        if not tracking:
            return jsonify({
                "status": "error",
                "message": f"Tracking with ID {tracking_id} not found"
            }), 404
            
        # Delete tracking
        result = MovementTrackingModel.delete_tracking(tracking_id)
        
        if result['success']:
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

@tracking_bp.route('/report/<report_id>', methods=['DELETE'])
def delete_tracking_by_report(report_id):
    """Delete tracking by report ID."""
    try:
        # Check if tracking exists
        tracking = MovementTrackingModel.get_tracking_by_report_id(report_id)
        
        if not tracking:
            return jsonify({
                "status": "error",
                "message": f"No tracking found for report ID {report_id}"
            }), 404
            
        # Delete tracking
        result = MovementTrackingModel.delete_tracking_by_report_id(report_id)
        
        if result['success']:
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

@tracking_bp.route('/create-all', methods=['POST'])
def create_all_tracking():
    """Create tracking entries for all reports that have movement data."""
    try:
        result = MovementTrackingModel.create_all_from_reports()
        
        return jsonify({
            "status": "success",
            "message": f"Created {result['created_count']} tracking entries, skipped {result['skipped_count']} existing entries"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error creating tracking entries: {str(e)}"
        }), 500

@tracking_bp.route('/update-all', methods=['PUT'])
def update_all_tracking():
    """Update all tracking entries from their reports."""
    try:
        result = MovementTrackingModel.update_all_from_reports()
        
        return jsonify({
            "status": "success",
            "message": f"Updated {result['updated_count']} tracking entries, {result['not_found_count']} reports not found"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating tracking entries: {str(e)}"
        }), 500

@tracking_bp.route('/clear-all', methods=['DELETE'])
def clear_all_tracking():
    """Delete all tracking entries."""
    try:
        result = MovementTrackingModel.clear_all_tracking()
        
        return jsonify({
            "status": "success",
            "message": f"Deleted {result['deleted_count']} tracking entries"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error clearing tracking entries: {str(e)}"
        }), 500 