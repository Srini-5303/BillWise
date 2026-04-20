from __future__ import annotations

from pydantic import BaseModel, Field


class QualityAssessment(BaseModel):
    blur_score: float | None = None
    contrast_score: float | None = None
    brightness_score: float | None = None
    estimated_skew_deg: float | None = None
    issues: list[str] = Field(default_factory=list)


class VariantScore(BaseModel):
    name: str
    path: str
    ocr_line_count: int = 0
    ocr_token_count: int = 0
    avg_confidence: float = 0.0
    keyword_hits: int = 0
    overall_score: float = 0.0


class PreprocessingResult(BaseModel):
    original_path: str
    quality: QualityAssessment
    variants: list[VariantScore] = Field(default_factory=list)
    selected_ocr_variant: str
    selected_vlm_variant: str
    selected_ocr_path: str
    selected_vlm_path: str
    performed: bool = False