import os
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

class Database:
    db = None
    client = None

    @staticmethod
    async def connect():
        try:
            MONGO_URI = os.getenv("MONGO_URL")
            MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
            print(f"Connecting to MongoDB: {MONGO_URI}")

            Database.client = AsyncIOMotorClient(
                MONGO_URI, tls=True, tlsAllowInvalidCertificates=True
            )
            Database.db = Database.client[MONGO_DB_NAME]
            print("MongoDB connected.")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise e

    @staticmethod
    async def disconnect():
        try:
            if Database.client:
                Database.client.close()
                print("Database disconnected.")
        except Exception as e:
            print(f"Failed to disconnect from MongoDB: {e}")
