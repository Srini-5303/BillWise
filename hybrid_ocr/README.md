# Receipt KIE Prototype

Prototype for grocery receipt understanding using:

- PaddleOCR for OCR, bounding boxes, and OCR confidence
- LayoutLMv3 fine-tuned on WildReceipt for token classification
- Regex/rule-based post-processing for payment fields and cleanup

## Planned Output

For each extracted field:
- value
- confidence
- bounding box

## Modules

- `ocr/` -> OCR and preprocessing
- `kie/` -> LayoutLMv3 inference and label post-processing
- `extract/` -> field merging, regex extraction, validation
- `output/` -> final JSON formatting and visualization
- `app/` -> app entrypoint and schemas