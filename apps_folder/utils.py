from langchain_openai import ChatOpenAI
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import Dict, Any
import numpy as np
from apps_folder.models import TestCases
import os
import requests
import json
from bson import ObjectId
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings
from langserve import RemoteRunnable
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from pymongo import MongoClient
from apps_folder.constants import SYSTEM_PROMPT_CONVERT_TEXTCASES_TO_OP, SYSTEM_PROMPT_CONVERT_USER_TEXT_TO_TESTCASES
from apps_folder.database.db import Database

load_dotenv()

async def convert_numpy_types(data):
    if isinstance(data, np.bool_):  # Convert numpy bools to Python bools
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to lists
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    else:
        return data

async def generate_test_cases(agent_id: str, user_input: str) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{input}")])
    input_data = {
        "system": SYSTEM_PROMPT_CONVERT_USER_TEXT_TO_TESTCASES,
        "input": user_input
    }
    structured_llm = ChatOpenAI(model="gpt-4o").with_structured_output(TestCases)
    few_shot_structured_llm = prompt | structured_llm
    response = few_shot_structured_llm.invoke(input_data)
    return response

async def evaluate_test_cases(payload: dict, id):
    agent_id = id  # Use agent_id from the payload

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        model_kwargs={"response_format": {"type": "json_object"}},
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_CONVERT_TEXTCASES_TO_OP),
            ("human", "{input}"),
        ]
    )

    # Ensure Database connection
    db = Database.db
    cursor_bot_config = db['bot_configurations']
    cursor_faq_kbs = db['faq_kbs']
    cursor_bot_kb_mappings = db['bot_kb_mappings']

    # Fetch bot and KB details from MongoDB
    bot_config_details = await cursor_bot_config.find_one({"bot_id": ObjectId(agent_id)}, {})
    print(f"Bot Config Details: {bot_config_details}")  # Debugging line

    bot_details = await cursor_faq_kbs.find_one({"_id": ObjectId(agent_id)}, {})
    print(f"Bot Details: {bot_details}")  # Debugging line

    connected_kbs_cursor = cursor_bot_kb_mappings.find({"id_bot": ObjectId(agent_id)})
    kb_response = await connected_kbs_cursor.to_list(length=None) 
    kbs = [str(doc['id_kb']) for doc in kb_response]
    print(f"Connected KBs Response: {kb_response}")

    # Initialize Pinecone and VectorStore
    pinecone = Pinecone()
    vectorstore = PineconeVectorStore(
        index_name=bot_config_details['ai_config']['kb_configuration']['index_name'],
        embedding=embeddings,
        namespace=bot_config_details['ai_config']['kb_configuration']['namespace'],
        pinecone_api_key=os.environ.get("PINECONE_API_KEY")
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={'k': 5, 'filter': {"id_kb": {"$in": kbs}}}
    )

    # Initialize the remote runnable
    remote_runnable = RemoteRunnable("https://devbot.indemn.ai/chat")

    # Function to check semantic similarity
    async def check_semantic_similarity(expected_response, bot_response):
        try:
            chain = prompt | llm | SimpleJsonOutputParser()
            res = chain.invoke({
                "input": f"expected_response: {expected_response} bot_response: {bot_response}"
            })
            return res.get('matched', False)
        except Exception as e:
            return False

    # Function to evaluate RAGAS for QnA type
    async def evaluate_ragas(question, answer, contexts, threshold):
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy]
        )

        df = result.to_pandas()
        relevancy_score = df['answer_relevancy'].iloc[0]
        return relevancy_score >= threshold, relevancy_score

    # Function to create a new session
    async def create_session():
        url = f"{os.environ.get('MIDDLEWARE_URL')}/conversations"
        payload = {
            "bot_id": agent_id,
            "isTestMode": True
        }

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200 and response.json().get("ok"):
                return response.json().get("data", {}).get("session_id")
            else:
                return None
        except Exception as e:
            return None

    # Function to evaluate FLOW type test cases
    async def evaluate_flow(test_case, session_id):
        test_case['test_result'] = {"steps": []}

        for step in test_case['steps']:
            user_input = step['user_input']
            expected_response = step['expected_response']

            # Invoke remote_runnable
            response = remote_runnable.invoke({
                "input": user_input,
                "bot_details": {
                    "bot_id": {
                        "_id": agent_id,
                        "id_organization": bot_details['id_organization'],
                        "connected_kbs": kbs
                    },
                },
                "init_parameters": {},
                "session_id": session_id,
            })

            # Check semantic similarity asynchronously
            step_result = {
                "step": step['step'],
                "user_input": user_input,
                "expected_response": expected_response,
                "bot_response": response,
                "matched": await check_semantic_similarity(expected_response, response)
            }

            if not step_result['matched']:
                test_case['test_result']['steps'].append({
                    "result": "Failed",
                    **step_result
                })
                test_case['test_result']['result'] = "Failed"
                return False

            test_case['test_result']['steps'].append({
                "result": "Passed",
                **step_result
            })

        test_case['test_result']['result'] = "Passed"
        return True

    # Function to evaluate QnA type test cases
    async def evaluate_qna(test_case, session_id):
        test_case['test_result'] = {"steps": []}

        for step in test_case['steps']:
            question = step['user_input']
            expected_answer = step['expected_response']

            # Invoke remote_runnable
            answer = remote_runnable.invoke({
                "input": question,
                "bot_details": {
                    "bot_id": {
                        "_id": agent_id,
                        "id_organization": bot_details['id_organization'],
                        "connected_kbs": kbs
                    },
                },
                "init_parameters": {},
                "session_id": session_id,
            })

            # Retrieve contexts asynchronously
            contexts = [doc.page_content for doc in retriever.get_relevant_documents(question)]
            threshold = test_case['success_criteria']['threshold']
            relevancy_passed, relevancy_score = await evaluate_ragas(question, answer, contexts, threshold)

            step_result = {
                "question": question,
                "expected_response": expected_answer,
                "bot_response": answer,
                "contexts": contexts,
                "accuracy_score": relevancy_score,
                "matched": relevancy_passed and (expected_answer == answer)
            }

            if not step_result['matched']:
                test_case['test_result']['steps'].append({
                    "result": "Failed",
                    **step_result
                })
                test_case['test_result']['result'] = "Failed"
                return False

            test_case['test_result']['steps'].append({
                "result": "Passed",
                **step_result
            })

        test_case['test_result']['result'] = "Passed"
        return True

    # Loop through the test cases in the payload
    for test_case in payload['test_cases']:
        session_id = await create_session()  # Add await here
        if not session_id:
            continue

        if test_case['type'] == "FLOW":
            await evaluate_flow(test_case, session_id)  # Add await here
        elif test_case['type'] == "QnA":
            await evaluate_qna(test_case, session_id)  # Add await here

    return payload
