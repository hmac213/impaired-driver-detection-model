from flask import Blueprint, jsonify, request
from backend.models.reports import ReportsModel

reports_bp = Blueprint('reports', __name__)

@reports_bp.route('/', methods=['GET'])
def get_all_reports():
    """Get all reports with optional filtering."""
    try:
        # Get query parameters
        status = request.args.get('status')
        min_probability = request.args.get('min_probability')
        descriptor = request.args.get('descriptor')
        
        # Build query
        query = {}
        
        if status is not None:
            if status in ["null", "safe", "impaired"]:
                query['confirm_bool'] = status
            else:
                return jsonify({
                    "status": "error",
                    "message": "Invalid status parameter. Must be 'null', 'safe', or 'impaired'."
                }), 400
            
        if min_probability is not None:
            query['probability'] = {'$gte': float(min_probability)}
            
        if descriptor is not None:
            query['descriptor'] = descriptor
            
        # Get reports
        reports = ReportsModel.get_reports(query)
        
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

@reports_bp.route('/', methods=['POST'])
def create_new_report():
    """Create a new report."""
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
                
        # Create report
        result = ReportsModel.create_report(data)
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": "Report created successfully",
                "report_id": result['report_id']
            })
        else:
            return jsonify({
                "status": "error",
                "message": result.get('error', 'Failed to create report')
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error creating report: {str(e)}"
        }), 500

@reports_bp.route('/<report_id>', methods=['GET'])
def get_single_report(report_id):
    """Get a specific report by ID."""
    try:
        report = ReportsModel.get_report_by_id(report_id)
        
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

@reports_bp.route('/<report_id>', methods=['PUT'])
def update_existing_report(report_id):
    """Update a specific report."""
    try:
        data = request.json
        
        # Verify report exists
        existing_report = ReportsModel.get_report_by_id(report_id)
        
        if not existing_report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Extract only allowed fields
        update_data = {}
        for field in ['latitude', 'longitude', 'probability', 'descriptor', 'confirm_bool']:
            if field in data:
                update_data[field] = data[field]
                
        # Update report
        result = ReportsModel.update_report(report_id, update_data)
        
        if result['success']:
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

@reports_bp.route('/<report_id>', methods=['DELETE'])
def delete_existing_report(report_id):
    """Delete a specific report."""
    try:
        # Verify report exists
        existing_report = ReportsModel.get_report_by_id(report_id)
        
        if not existing_report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Delete report
        result = ReportsModel.delete_report(report_id)
        
        if result['success']:
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

@reports_bp.route('/confirm/<report_id>', methods=['PUT'])
def confirm_existing_report(report_id):
    """Mark a report as impaired."""
    try:
        # Verify report exists
        existing_report = ReportsModel.get_report_by_id(report_id)
        
        if not existing_report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Mark as impaired (for backward compatibility)
        result = ReportsModel.mark_as_impaired(report_id)
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": "Report marked as impaired"
            })
        else:
            return jsonify({
                "status": "error",
                "message": result.get('error', "Failed to update report")
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error confirming report: {str(e)}"
        }), 500

@reports_bp.route('/stats', methods=['GET'])
def get_report_statistics():
    """Get statistics about reports."""
    try:
        stats = ReportsModel.get_stats()
        
        return jsonify({
            "status": "success",
            **stats
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving stats: {str(e)}"
        }), 500

@reports_bp.route('/samples', methods=['POST'])
def create_sample_data():
    """Create sample reports for testing."""
    try:
        result = ReportsModel.create_sample_reports()
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": f"Created {result['count']} sample reports"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to create sample reports"
            }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error creating sample reports: {str(e)}"
        }), 500

@reports_bp.route('/status/<report_id>', methods=['GET'])
def get_report_status(report_id):
    """Get just the status of a specific report."""
    try:
        report = ReportsModel.get_report_by_id(report_id)
        
        if not report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Return only the status information
        status_info = {
            "report_id": report["report_id"],
            "confirm_bool": report["confirm_bool"],
            "probability": report["probability"],
            "descriptor": report["descriptor"],
            "created_at": report["created_at"],
            "updated_at": report["updated_at"]
        }
        
        return jsonify({
            "status": "success",
            "report_status": status_info
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving report status: {str(e)}"
        }), 500

@reports_bp.route('/mark-safe/<report_id>', methods=['PUT'])
def mark_report_as_safe(report_id):
    """Mark a report as safe."""
    try:
        # Verify report exists
        existing_report = ReportsModel.get_report_by_id(report_id)
        
        if not existing_report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Mark as safe
        result = ReportsModel.mark_as_safe(report_id)
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": "Report marked as safe"
            })
        else:
            return jsonify({
                "status": "error",
                "message": result.get('error', "Failed to update report")
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error marking report as safe: {str(e)}"
        }), 500

@reports_bp.route('/mark-impaired/<report_id>', methods=['PUT'])
def mark_report_as_impaired(report_id):
    """Mark a report as impaired."""
    try:
        # Verify report exists
        existing_report = ReportsModel.get_report_by_id(report_id)
        
        if not existing_report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Mark as impaired
        result = ReportsModel.mark_as_impaired(report_id)
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": "Report marked as impaired"
            })
        else:
            return jsonify({
                "status": "error",
                "message": result.get('error', "Failed to update report")
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error marking report as impaired: {str(e)}"
        }), 500

@reports_bp.route('/mark-null/<report_id>', methods=['PUT'])
def mark_report_as_null(report_id):
    """Reset a report status to null."""
    try:
        # Verify report exists
        existing_report = ReportsModel.get_report_by_id(report_id)
        
        if not existing_report:
            return jsonify({
                "status": "error",
                "message": f"Report with ID {report_id} not found"
            }), 404
            
        # Reset to null
        result = ReportsModel.set_report_status(report_id, "null")
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": "Report status reset to null"
            })
        else:
            return jsonify({
                "status": "error",
                "message": result.get('error', "Failed to update report")
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error resetting report status: {str(e)}"
        }), 500

@reports_bp.route('/update-existing-reports', methods=['POST'])
def update_legacy_reports():
    """Update existing reports to use new status values."""
    try:
        result = ReportsModel.update_existing_reports()
        
        return jsonify({
            "status": "success",
            "message": f"Updated {result['updated_count']} reports to use the new status values"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating reports: {str(e)}"
        }), 500

@reports_bp.route('/reset-with-samples', methods=['POST'])
def reset_with_samples():
    """Clear all reports and create new sample reports."""
    try:
        # Clear all reports
        clear_result = ReportsModel.clear_all_reports()
        
        # Create sample reports
        samples_result = ReportsModel.create_sample_reports()
        
        return jsonify({
            "status": "success",
            "message": f"Cleared {clear_result['deleted_count']} reports and created {samples_result['count']} sample reports"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error resetting reports: {str(e)}"
        }), 500 