from flask import Blueprint, jsonify, request

camera_bp = Blueprint('camera', __name__)

@camera_bp.route('/', methods=['GET'])
def get_all_cameras():
    """Get all camera data."""
    return jsonify({
        "status": "success",
        "message": "Camera routes will be implemented here"
    })

# You can implement additional camera-related endpoints here 