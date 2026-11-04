from fastapi import APIRouter, HTTPException

from backend.app.schemas.analysis import AnalysisResponse, AnalystRequest

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_request(payload: AnalystRequest):
    question = payload.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if question.lower() in {"n/a", "na", "none", "null"}:
        raise HTTPException(
            status_code=400,
            detail="Question is not valid for analysis. Please provide a meaningful question.",
        )

    return {
        "question": question,
        "file_name": payload.file_name,
    }
