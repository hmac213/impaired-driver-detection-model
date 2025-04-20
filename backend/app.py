from flask import Flask
from flask_cors import CORS
import sys
import os

# Add parent directory to path to make backend imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.routes.reports import reports_bp
from backend.routes.camera import camera_bp
from backend.routes.detection import detection_bp
from backend.routes.auth import auth_bp
from backend.routes.tracking import tracking_bp

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    # Register blueprints
    app.register_blueprint(reports_bp, url_prefix='/api/reports')
    app.register_blueprint(camera_bp, url_prefix='/api/camera')
    app.register_blueprint(detection_bp, url_prefix='/api/detection')
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(tracking_bp, url_prefix='/api/tracking')
    
    @app.route('/')
    def index():
        return {
            'status': 'success',
            'message': 'Impaired Driver Detection API is running',
            'endpoints': {
                'reports': '/api/reports',
                'camera': '/api/camera',
                'detection': '/api/detection',
                'auth': '/api/auth',
                'tracking': '/api/tracking'
            }
        }
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000) 