from pymongo import MongoClient
import datetime
import uuid
import bcrypt
import json
from bson import json_util

# MongoDB connection string
MONGO_URI = "mongodb+srv://Temp:hack@cluster0.lhazyyp.mongodb.net/DriverData?retryWrites=true&w=majority&appName=Cluster0"

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

def main():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client["DriverData"]
        rbac_collection = db["rbac"]
        
        print("Connected to MongoDB successfully!")
        
        # Create indexes
        rbac_collection.create_index("email", unique=True)
        rbac_collection.create_index("user_id")
        print("Created indexes for email and user_id")
        
        # Clear existing users
        clear_result = rbac_collection.delete_many({})
        print(f"Cleared {clear_result.deleted_count} existing users")
        
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
        print(f"Created {len(result.inserted_ids)} sample users")
        
        # Verify users were created
        users = list(rbac_collection.find({}, {'_id': 0, 'password': 0}))
        
        print("\nUsers in database:")
        for user in users:
            print(json.dumps(parse_json(user), indent=2))
            
        print("\nRBAC collection initialized successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 