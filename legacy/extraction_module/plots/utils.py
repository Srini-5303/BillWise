import json
from pathlib import Path

# Paths
REPORTS_DIR = Path("assets/dataset/evaluation_reports")
PREDICTIONS_DIR = Path("assets/dataset/predictions")
IMAGES_DIR = Path("assets/dataset/receipts")
ANNOTATIONS_DIR = Path("assets/dataset/annotation_feedback")

# Consistent color palette
METHOD_COLORS = {
    "Prototype": "#1f77b4", # Blue
    "Groq": "#ff7f0e",      # Orange
    "Hybrid": "#2ca02c"     # Green
}

METHOD_MAPPING = {
    "prototype_layoutlm": "Prototype",
    "groq_vlm": "Groq",
    "hybrid_proto_groq": "Hybrid"
}

def load_summary(method_name: str) -> dict:
    path = REPORTS_DIR / f"{method_name}_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_per_receipt(method_name: str) -> list:
    path = REPORTS_DIR / f"{method_name}_per_receipt.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def get_store_from_id(receipt_id: str) -> str:
    if receipt_id.startswith("costco"):
        return "Costco"
    elif receipt_id.startswith("target"):
        return "Target"
    elif receipt_id.startswith("traderjoes"):
        return "Trader Joe's"
    return "Unknown"
