from flask import Flask, jsonify, request
from pymongo import MongoClient
import datetime
import uuid
import bcrypt
import json
from bson import json_util

app = Flask(__name__)

# MongoDB connection string
MONGO_URI = "mongodb+srv://Temp:hack@cluster0.lhazyyp.mongodb.net/DriverData?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["DriverData"]
rbac_collection = db["rbac"]

# Valid role values
VALID_ROLES = ["law enforcement", "community"]

def parse_json(data):
    """Convert MongoDB results to JSON serializable format."""
    return json.loads(json_util.dumps(data))

def hash_password(password):
    """Hash a password using bcrypt."""
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')  # Store as string

def check_password(stored_hash, provided_password):
    """Check if the provided password matches the stored hash."""
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_hash.encode('utf-8'))

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

@app.route("/users", methods=["POST"])
def create_user():
    """Create a new user in the RBAC collection."""
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
        
        # Check if email already exists
        existing_user = rbac_collection.find_one({"email": data['email']})
        if existing_user:
            return jsonify({
                "status": "error",
                "message": "Email already registered"
            }), 409
        
        # Generate a unique user_id
        user_id = str(uuid.uuid4())
        
        # Hash the password
        hashed_password = hash_password(data['password'])
        
        # Create user document
        user = {
            "user_id": user_id,
            "name": data['name'],
            "email": data['email'],
            "password": hashed_password,
            "role": data['role'],
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        # Insert user into MongoDB
        result = rbac_collection.insert_one(user)
        
        if result.acknowledged:
            return jsonify({
                "status": "success",
                "message": "User created successfully",
                "user_id": user_id
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to create user"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error creating user: {str(e)}"
        }), 500

@app.route("/users", methods=["GET"])
def get_users():
    """Get all users (excluding password hashes)."""
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
            
        # Query MongoDB for users (excluding password)
        users = list(rbac_collection.find(query, {'_id': 0, 'password': 0}))
        
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

@app.route("/users/<user_id>", methods=["GET"])
def get_user(user_id):
    """Get a specific user by ID (excluding password hash)."""
    try:
        # Query MongoDB for the user
        user = rbac_collection.find_one({"user_id": user_id}, {'_id': 0, 'password': 0})
        
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

@app.route("/users/<user_id>", methods=["PUT"])
def update_user(user_id):
    """Update a specific user."""
    try:
        data = request.json
        
        # Query MongoDB for the user
        existing_user = rbac_collection.find_one({"user_id": user_id})
        
        if not existing_user:
            return jsonify({
                "status": "error",
                "message": f"User with ID {user_id} not found"
            }), 404
            
        # Update fields
        update_data = {
            "updated_at": datetime.datetime.utcnow()
        }
        
        # Check for email uniqueness if changing email
        if 'email' in data and data['email'] != existing_user['email']:
            email_check = rbac_collection.find_one({"email": data['email']})
            if email_check:
                return jsonify({
                    "status": "error",
                    "message": "Email already registered to another user"
                }), 409
            update_data['email'] = data['email']
            
        # Update name if provided
        if 'name' in data:
            update_data['name'] = data['name']
            
        # Update role if provided
        if 'role' in data:
            if data['role'] not in VALID_ROLES:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}"
                }), 400
            update_data['role'] = data['role']
            
        # Update password if provided
        if 'password' in data:
            update_data['password'] = hash_password(data['password'])
                
        # Update the user
        result = rbac_collection.update_one(
            {"user_id": user_id},
            {"$set": update_data}
        )
        
        if result.modified_count == 1:
            return jsonify({
                "status": "success",
                "message": "User updated successfully"
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

@app.route("/users/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    """Delete a specific user."""
    try:
        # Query MongoDB for the user
        existing_user = rbac_collection.find_one({"user_id": user_id})
        
        if not existing_user:
            return jsonify({
                "status": "error",
                "message": f"User with ID {user_id} not found"
            }), 404
            
        # Delete the user
        result = rbac_collection.delete_one({"user_id": user_id})
        
        if result.deleted_count == 1:
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

@app.route("/login", methods=["POST"])
def login():
    """Authenticate a user by email and password."""
    try:
        data = request.json
        
        # Validate required fields
        if 'email' not in data or 'password' not in data:
            return jsonify({
                "status": "error",
                "message": "Email and password are required"
            }), 400
        
        # Find user by email
        user = rbac_collection.find_one({"email": data['email']})
        
        if not user:
            return jsonify({
                "status": "error",
                "message": "Invalid email or password"
            }), 401
        
        # Check password
        if check_password(user['password'], data['password']):
            # Return user info without password
            user_info = {k: v for k, v in user.items() if k != 'password' and k != '_id'}
            
            return jsonify({
                "status": "success",
                "message": "Login successful",
                "user": user_info
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid email or password"
            }), 401
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error during login: {str(e)}"
        }), 500

@app.route("/create-sample-users", methods=["POST"])
def create_sample_users():
    """Create sample users for testing."""
    try:
        sample_users = [
            {
                "user_id": str(uuid.uuid4()),
                "name": "Officer John Smith",
                "email": "john.smith@police.gov",
                "password": hash_password("secure123"),
                "role": "law enforcement",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            },
            {
                "user_id": str(uuid.uuid4()),
                "name": "Jane Citizen",
                "email": "jane@citizen.org",
                "password": hash_password("community456"),
                "role": "community",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            },
            {
                "user_id": str(uuid.uuid4()),
                "name": "Sergeant Mike Johnson",
                "email": "mike.johnson@police.gov",
                "password": hash_password("police789"),
                "role": "law enforcement",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
        ]
        
        # Insert sample users
        result = rbac_collection.insert_many(sample_users)
        
        return jsonify({
            "status": "success",
            "message": f"Created {len(result.inserted_ids)} sample users"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error creating sample users: {str(e)}"
        }), 500

@app.route("/reset-users", methods=["POST"])
def reset_users():
    """Clear all users and create new sample users."""
    try:
        # Clear all users
        clear_result = rbac_collection.delete_many({})
        
        # Create sample users
        sample_users = [
            {
                "user_id": str(uuid.uuid4()),
                "name": "Officer John Smith",
                "email": "john.smith@police.gov",
                "password": hash_password("secure123"),
                "role": "law enforcement",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            },
            {
                "user_id": str(uuid.uuid4()),
                "name": "Jane Citizen",
                "email": "jane@citizen.org",
                "password": hash_password("community456"),
                "role": "community",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            },
            {
                "user_id": str(uuid.uuid4()),
                "name": "Sergeant Mike Johnson",
                "email": "mike.johnson@police.gov",
                "password": hash_password("police789"),
                "role": "law enforcement",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
        ]
        
        # Insert sample users
        result = rbac_collection.insert_many(sample_users)
        
        return jsonify({
            "status": "success",
            "message": f"Cleared {clear_result.deleted_count} users and created {len(result.inserted_ids)} sample users"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error resetting users: {str(e)}"
        }), 500

if __name__ == "__main__":
    # Create index for email uniqueness
    rbac_collection.create_index("email", unique=True)
    
    # Create index for user_id
    rbac_collection.create_index("user_id")
    
    app.run(debug=True, port=5002) 