import logging
from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGODB_URI, MONGODB_DB_NAME

logger = logging.getLogger(__name__)

class DatabaseProvider:
    client: AsyncIOMotorClient = None
    db = None

db_provider = DatabaseProvider()

async def connect_to_mongo():
    try:
        logger.info("Connecting to MongoDB...")
        # Add serverSelectionTimeoutMS for quick timeout if IP isn't whitelisted
        db_provider.client = AsyncIOMotorClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            uuidRepresentation='standard'
        )
        db_provider.db = db_provider.client[MONGODB_DB_NAME]
        
        # Ping
        await db_provider.client.admin.command('ping')
        logger.info(f"Connected to MongoDB. Database: {MONGODB_DB_NAME}")
    except Exception as e:
        logger.error(f"Could not connect to MongoDB: {e}")

async def close_mongo_connection():
    if db_provider.client:
        logger.info("Closing MongoDB connection...")
        db_provider.client.close()
        logger.info("MongoDB connection closed.")

def get_db():
    if db_provider.db is None:
        raise RuntimeError("Database is not initialized. Call connect_to_mongo first.")
    return db_provider.db
