import os
from typing import Union
from bson import ObjectId
from fastapi import HTTPException, FastAPI
from fastapi.concurrency import run_in_threadpool
from ..utils import convert_numpy_types, generate_test_cases, evaluate_test_cases
from apps_folder.database.db import Database


class EvaluateService:
    def __init__(self, payload: dict):
        self.agent_id = payload.get("agent_id")
        self.user_input = payload.get("user_input")

    async def validate_agent_id(self) -> bool:
        
        try:
            # Access the MongoDB connection
            #db = Database.db['faq_kbs']
            if Database.db is None:
                print("Database is not connected. Connecting now...")
                await Database.connect()
            
            db = Database.db['faq_kbs']


            # Find the agent ID in the collection
            agent_exists = await db.find_one({"_id": ObjectId(self.agent_id)})

            if not agent_exists:
                raise ValueError("Invalid agent ID provided.")
            return True

        except Exception as e:
            raise ValueError(f"Error validating agent ID: {str(e)}")

    async def start_process(self) -> dict:
        """
        Orchestrates the overall process of validation and LLM invocation.
        """
        await self.validate_agent_id()  # Await validation
        response =  await generate_test_cases(self.agent_id, self.user_input)
        evaluation_results = await evaluate_test_cases(response, self.agent_id)
        converted_response = await convert_numpy_types(evaluation_results)  
        
        return converted_response
