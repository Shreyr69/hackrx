from pydantic import BaseModel, Field
from typing import List


class RunRequest(BaseModel):
    documents: str = Field(..., description="Blob URL to a document (PDF/DOCX/Email)")
    questions: List[str] = Field(..., description="List of user questions")


class RunResponse(BaseModel):
    answers: List[str]
