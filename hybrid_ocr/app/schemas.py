from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class OCRToken(BaseModel):
    id: int
    text: str
    bbox: List[int] = Field(..., min_length=4, max_length=4)
    poly: Optional[List[List[int]]] = None
    ocr_confidence: float
    det_confidence: Optional[float] = None


class ExtractedField(BaseModel):
    field_name: str
    value: str
    confidence: float
    bbox: List[int] = Field(..., min_length=4, max_length=4)
    source: Literal["layoutlm", "regex+ocr", "ocr_only"]
    ocr_confidence: Optional[float] = None
    model_confidence: Optional[float] = None
    token_ids: List[int] = []


class LineItemField(BaseModel):
    value: str
    confidence: float
    bbox: List[int] = Field(..., min_length=4, max_length=4)


class LineItem(BaseModel):
    description: Optional[LineItemField] = None
    quantity: Optional[LineItemField] = None
    price: Optional[LineItemField] = None


class ReceiptResult(BaseModel):
    image_path: str
    image_width: int
    image_height: int
    fields: List[ExtractedField]
    items: List[LineItem]
    raw_ocr: List[OCRToken]