import datetime
import uuid
import bcrypt
from backend.config.database import rbac_collection

class RbacModel:
    """Model for handling user authentication and role-based access control."""
    
    @staticmethod
    def hash_password(password):
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')  # Store as string

    @staticmethod
    def check_password(stored_hash, provided_password):
        """Check if the provided password matches the stored hash."""
        return bcrypt.checkpw(provided_password.encode('utf-8'), stored_hash.encode('utf-8'))
    
    @staticmethod
    def create_user(data):
        """Create a new user in the database."""
        # Check if email already exists
        existing_user = rbac_collection.find_one({"email": data['email']})
        if existing_user:
            return {"success": False, "error": "Email already registered"}
        
        # Generate a unique user_id
        user_id = str(uuid.uuid4())
        
        # Hash the password
        hashed_password = RbacModel.hash_password(data['password'])
        
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
            return {"success": True, "user_id": user_id}
        else:
            return {"success": False, "error": "Failed to create user"}
    
    @staticmethod
    def get_users(query=None):
        """Get users based on query parameters."""
        if query is None:
            query = {}
            
        # Query MongoDB for users (excluding password)
        users = list(rbac_collection.find(query, {'_id': 0, 'password': 0}))
        return users
    
    @staticmethod
    def get_user_by_id(user_id):
        """Get a specific user by ID."""
        user = rbac_collection.find_one({"user_id": user_id}, {'_id': 0, 'password': 0})
        return user
    
    @staticmethod
    def get_user_by_email(email):
        """Get a specific user by email."""
        user = rbac_collection.find_one({"email": email}, {'_id': 0})
        return user
    
    @staticmethod
    def update_user(user_id, update_data):
        """Update a specific user."""
        # Add updated timestamp
        update_data["updated_at"] = datetime.datetime.utcnow()
        
        # Hash password if provided
        if 'password' in update_data:
            update_data['password'] = RbacModel.hash_password(update_data['password'])
        
        # Update the user
        result = rbac_collection.update_one(
            {"user_id": user_id},
            {"$set": update_data}
        )
        
        return {
            "success": result.modified_count == 1,
            "matched": result.matched_count > 0
        }
    
    @staticmethod
    def delete_user(user_id):
        """Delete a specific user."""
        result = rbac_collection.delete_one({"user_id": user_id})
        
        return {
            "success": result.deleted_count == 1,
            "deleted": result.deleted_count > 0
        }
    
    @staticmethod
    def authenticate(email, password):
        """Authenticate a user with email and password."""
        user = rbac_collection.find_one({"email": email})
        
        if not user:
            return {"success": False, "error": "Invalid email or password"}
        
        if RbacModel.check_password(user['password'], password):
            # Return user info without password and _id
            user_info = {k: v for k, v in user.items() if k != 'password' and k != '_id'}
            return {"success": True, "user": user_info}
        else:
            return {"success": False, "error": "Invalid email or password"}
    
    @staticmethod
    def is_authorized(user_id, required_role):
        """Check if a user has the required role."""
        user = rbac_collection.find_one({"user_id": user_id})
        
        if not user:
            return False
        
        return user['role'] == required_role
    
    @staticmethod
    def create_sample_users():
        """Create sample users for testing."""
        sample_users = [
            {
                "user_id": str(uuid.uuid4()),
                "name": "Officer John Smith",
                "email": "john.smith@police.gov",
                "password": RbacModel.hash_password("secure123"),
                "role": "law enforcement",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            },
            {
                "user_id": str(uuid.uuid4()),
                "name": "Jane Citizen",
                "email": "jane@citizen.org",
                "password": RbacModel.hash_password("community456"),
                "role": "community",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            },
            {
                "user_id": str(uuid.uuid4()),
                "name": "Sergeant Mike Johnson",
                "email": "mike.johnson@police.gov",
                "password": RbacModel.hash_password("police789"),
                "role": "law enforcement",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
        ]
        
        # Insert sample users
        result = rbac_collection.insert_many(sample_users)
        
        return {
            "success": len(result.inserted_ids) > 0,
            "count": len(result.inserted_ids)
        }
        
    @staticmethod
    def clear_all_users():
        """Delete all users from the collection."""
        result = rbac_collection.delete_many({})
        return {
            "success": True,
            "deleted_count": result.deleted_count
        }
    
    @staticmethod
    def reset_with_samples():
        """Clear all users and create new sample users."""
        # Clear all users
        clear_result = RbacModel.clear_all_users()
        
        # Create sample users
        samples_result = RbacModel.create_sample_users()
        
        return {
            "success": samples_result['success'],
            "cleared": clear_result['deleted_count'],
            "created": samples_result['count']
        } 