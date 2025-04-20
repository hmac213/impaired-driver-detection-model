from flask import Blueprint, jsonify, request
from backend.models.rbac import RbacModel

auth_bp = Blueprint('auth', __name__)

# Valid role values
VALID_ROLES = ["law enforcement", "community"]

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['name', 'email', 'password', 'role']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "status": "error",
                    "message": f"Missing required field: {field}"
                }), 400
                
        # Validate role
        if data['role'] not in VALID_ROLES:
            return jsonify({
                "status": "error",
                "message": f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}"
            }), 400
        
        # Create user
        result = RbacModel.create_user(data)
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": "User registered successfully",
                "user_id": result['user_id']
            })
        else:
            return jsonify({
                "status": "error",
                "message": result.get('error', 'Failed to register user')
            }), 409 if 'already registered' in result.get('error', '') else 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error registering user: {str(e)}"
        }), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Authenticate a user."""
    try:
        data = request.json
        
        # Validate required fields
        if 'email' not in data or 'password' not in data:
            return jsonify({
                "status": "error",
                "message": "Email and password are required"
            }), 400
        
        # Authenticate user
        result = RbacModel.authenticate(data['email'], data['password'])
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": "Login successful",
                "user": result['user']
            })
        else:
            return jsonify({
                "status": "error",
                "message": result.get('error', 'Authentication failed')
            }), 401
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error during login: {str(e)}"
        }), 500

@auth_bp.route('/users', methods=['GET'])
def get_users():
    """Get all users with optional role filtering."""
    try:
        # Get query parameters
        role = request.args.get('role')
        
        # Build query
        query = {}
        if role is not None:
            if role in VALID_ROLES:
                query['role'] = role
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid role parameter. Must be one of: {', '.join(VALID_ROLES)}"
                }), 400
                
        # Get users
        users = RbacModel.get_users(query)
        
        return jsonify({
            "status": "success",
            "count": len(users),
            "users": users
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving users: {str(e)}"
        }), 500

@auth_bp.route('/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get a specific user by ID."""
    try:
        user = RbacModel.get_user_by_id(user_id)
        
        if not user:
            return jsonify({
                "status": "error",
                "message": f"User with ID {user_id} not found"
            }), 404
            
        return jsonify({
            "status": "success",
            "user": user
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving user: {str(e)}"
        }), 500

@auth_bp.route('/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    """Update a specific user."""
    try:
        data = request.json
        
        # Verify user exists
        existing_user = RbacModel.get_user_by_id(user_id)
        
        if not existing_user:
            return jsonify({
                "status": "error",
                "message": f"User with ID {user_id} not found"
            }), 404
            
        # Validate role if provided
        if 'role' in data and data['role'] not in VALID_ROLES:
            return jsonify({
                "status": "error",
                "message": f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}"
            }), 400
            
        # Check for email uniqueness if changing email
        if 'email' in data and data['email'] != existing_user['email']:
            email_check = RbacModel.get_user_by_email(data['email'])
            if email_check:
                return jsonify({
                    "status": "error",
                    "message": "Email already registered to another user"
                }), 409
                
        # Update user
        result = RbacModel.update_user(user_id, data)
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": "User updated successfully"
            })
        elif result['matched']:
            return jsonify({
                "status": "info",
                "message": "No changes made to user"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to update user"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating user: {str(e)}"
        }), 500

@auth_bp.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a specific user."""
    try:
        # Verify user exists
        existing_user = RbacModel.get_user_by_id(user_id)
        
        if not existing_user:
            return jsonify({
                "status": "error",
                "message": f"User with ID {user_id} not found"
            }), 404
            
        # Delete user
        result = RbacModel.delete_user(user_id)
        
        if result['success']:
            return jsonify({
                "status": "success",
                "message": "User deleted successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to delete user"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error deleting user: {str(e)}"
        }), 500

@auth_bp.route('/reset-with-samples', methods=['POST'])
def reset_with_samples():
    """Clear all users and create new sample users."""
    try:
        result = RbacModel.reset_with_samples()
        
        return jsonify({
            "status": "success",
            "message": f"Cleared {result['cleared']} users and created {result['created']} sample users"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error resetting users: {str(e)}"
        }), 500 