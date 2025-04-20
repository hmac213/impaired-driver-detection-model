from flask import Blueprint, jsonify, request

detection_bp = Blueprint('detection', __name__)

@detection_bp.route('/', methods=['GET'])
def get_detection_info():
    """Get detection system information."""
    return jsonify({
        "status": "success",
        "message": "Detection routes will be implemented here"
    })

# You can implement detection-related endpoints here 