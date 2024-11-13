from pydantic import BaseModel
from typing import List, Union, TypedDict

class Step(TypedDict):
    step: str
    user_input: str
    expected_response: str

class SuccessCriteria(TypedDict):
    threshold: float

class TestCase(TypedDict):
    description: str
    type: str
    steps: List[Step]
    success_criteria: Union[str, SuccessCriteria]
    failure_criteria: str

class TestCases(TypedDict):
    test_cases: List[TestCase]

# Input classes for FastAPI
class TestCasesInput(BaseModel):
    user_input: str
    agent_id: str
