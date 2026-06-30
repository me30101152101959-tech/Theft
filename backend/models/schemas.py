"""API Request / Response Schemas"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Any, List


class ManualPredictionRequest(BaseModel):
    customer_id: str = Field(..., description="Customer identifier")
    readings: List[float] = Field(..., min_items=2,
                                   description="Electricity consumption readings (any length > 1)")
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    strategy: str = Field("last_n", description="Preprocessing strategy when length != model length")


class ThresholdUpdateRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)


class CopilotRequest(BaseModel):
    question: str
    language: str = "en"
    customer_id: Optional[str] = None


class CustomerQuery(BaseModel):
    page: int = 1
    page_size: int = 50
    search: str = ""
    status_filter: str = ""
    sort_by: str = "risk_score"
    sort_dir: str = "desc"


class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: str = ""
    error: Optional[str] = None
