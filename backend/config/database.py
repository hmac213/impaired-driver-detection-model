from pymongo import MongoClient

# MongoDB connection string
MONGO_URI = "mongodb+srv://Temp:hack@cluster0.lhazyyp.mongodb.net/DriverData?retryWrites=true&w=majority&appName=Cluster0"

# Create MongoDB client instance
client = MongoClient(MONGO_URI)

# Database
db = client["DriverData"]

# Collections
reports_collection = db["reports"]
camera_collection = db["camera_db"]
rbac_collection = db["rbac"]
movement_tracking_collection = db["movement_tracking"]

# Ensure email uniqueness in rbac collection
rbac_collection.create_index("email", unique=True)

# Create index for user_id in rbac collection
rbac_collection.create_index("user_id")

# Create indexes for movement_tracking collection
movement_tracking_collection.create_index("tracking_id", unique=True)
movement_tracking_collection.create_index("report_id", unique=True) 