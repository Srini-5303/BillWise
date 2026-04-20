from pathlib import Path

from billwise.preprocessing.artifacts import save_preprocessing_artifact
from billwise.preprocessing.schemas import PreprocessingResult, QualityAssessment, VariantScore


def test_phaseP2_save_preprocessing_artifact(tmp_path, monkeypatch):
    from billwise.common import config as config_module

    class DummyPaths:
        processed_dir = tmp_path

    class DummyConfig:
        paths = DummyPaths()

    monkeypatch.setattr(config_module, "get_config", lambda: DummyConfig())

    result = PreprocessingResult(
        original_path="data/raw/abc.jpg",
        quality=QualityAssessment(
            blur_score=100.0,
            contrast_score=40.0,
            brightness_score=180.0,
            estimated_skew_deg=0.5,
            issues=[],
        ),
        variants=[
            VariantScore(
                name="original",
                path="data/raw/abc.jpg",
                ocr_line_count=10,
                ocr_token_count=25,
                avg_confidence=0.9,
                keyword_hits=2,
                overall_score=0.8,
            )
        ],
        selected_ocr_variant="original",
        selected_vlm_variant="original",
        selected_ocr_path="data/raw/abc.jpg",
        selected_vlm_path="data/raw/abc.jpg",
        performed=False,
    )

    artifact = save_preprocessing_artifact("rcpt_test", result)
    assert artifact.exists()
    assert artifact.name == "rcpt_test.preprocessing.json"