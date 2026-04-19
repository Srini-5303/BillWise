from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"

# OCR
USE_DOC_ORIENTATION_CLASSIFY = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False

# Model
LAYOUTLM_MODEL_NAME = "avurity/layoutlmv3-finetuned-wildreceipt"

# Inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYOUTLM_WORD_CHUNK_SIZE = 90
LAYOUTLM_WORD_OVERLAP = 15

# Output
DEBUG_DIR = ASSETS_DIR / "debug"
TEMP_DIR = ASSETS_DIR / "temp"

DEBUG_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
