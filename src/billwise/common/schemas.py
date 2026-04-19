from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ExtractedField(BaseModel):
    field_name: str
    field_value: str | None = None
    confidence: float | None = None
    bbox: BoundingBox | None = None
    source_model: str | None = None


class LineItem(BaseModel):
    item_id: str
    raw_name: str
    normalized_name: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    item_total: float | None = None
    item_confidence: float | None = None
    item_source: str | None = None


class CategoryPrediction(BaseModel):
    item_id: str
    predicted_category: str | None = None
    category_confidence: float | None = None
    top_k_scores: dict[str, float] = Field(default_factory=dict)
    categorizer_model: str | None = None
    needs_human_review: bool = False
    final_category: str | None = None


class ReceiptRecord(BaseModel):
    receipt_id: str
    image_path: str
    source: str = "local_upload"
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_status: str = "uploaded"
    review_status: str = "pending"
    vendor_name: str | None = None
    receipt_date: str | None = None
    receipt_time: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    payment_method: str | None = None
    card_last4: str | None = None
    receipt_number: str | None = None
    extraction_method: str | None = None
    requires_review: bool = False
    fields: list[ExtractedField] = Field(default_factory=list)
    items: list[LineItem] = Field(default_factory=list)


class ReviewChange(BaseModel):
    entity_type: str
    entity_id: str
    field_name: str
    old_value: Any = None
    new_value: Any = None
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)
    review_source: str = "ui"


class PipelineRun(BaseModel):
    run_id: str
    receipt_id: str
    stage: str
    status: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    error_message: str | None = None