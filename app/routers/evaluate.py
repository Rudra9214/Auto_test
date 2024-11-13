from fastapi import APIRouter, HTTPException
from .. import models
from ..services.ai_service import EvaluateService


router = APIRouter(prefix="/testcase", tags=["evaluate"])


@router.post("/", response_model=None)
async def evaluate_testcase(testcase: models.TestCasesInput):
    """
    Endpoint to evaluate test cases using the EvaluateService.
    """
    try:
        # Initialize the EvaluateService with payload and request
        service = EvaluateService(testcase.model_dump())

        # Generate the test case prompt using the service and request
        result = await service.start_process()

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}")
