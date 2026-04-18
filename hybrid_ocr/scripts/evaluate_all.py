from __future__ import annotations

import json
from pathlib import Path

from evaluation.loaders import load_gold_receipt
from evaluation.metrics import score_receipt, summarize_scores
from methods.groq_vlm import GroqVLMMethod


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    repo_root = Path(__file__).resolve().parent.parent

    dataset_dir = repo_root / "assets" / "dataset"
    receipts_dir = dataset_dir / "receipts"
    gold_dir = dataset_dir / "gold_labels"

    predictions_root = dataset_dir / "predictions"
    reports_root = dataset_dir / "evaluation_reports"

    method = GroqVLMMethod()

    receipt_paths = sorted(
        [p for p in receipts_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    )

    print(f"\n=== Evaluating {method.name} ===")

    rows = []
    method_pred_dir = predictions_root / method.name
    method_pred_dir.mkdir(parents=True, exist_ok=True)

    for image_path in receipt_paths:
        receipt_id = image_path.stem
        gold_path = gold_dir / f"{receipt_id}.json"

        if not gold_path.exists():
            print(f"Missing gold label for {receipt_id}, skipping")
            continue

        gold = load_gold_receipt(gold_path)

        try:
            pred = method.extract(str(image_path), receipt_id)
            metrics = score_receipt(pred, gold)

            save_json(method_pred_dir / f"{receipt_id}.json", pred.model_dump())

            rows.append(
                {
                    "receipt_id": receipt_id,
                    "image_file": image_path.name,
                    "metrics": metrics,
                }
            )

            print(f"{receipt_id}: overall={metrics['overall_score']:.4f}")

        except Exception as e:
            rows.append(
                {
                    "receipt_id": receipt_id,
                    "image_file": image_path.name,
                    "error": str(e),
                }
            )
            print(f"{receipt_id}: ERROR -> {e}")

    summary = summarize_scores(rows)

    save_json(reports_root / f"{method.name}_per_receipt.json", rows)
    save_json(reports_root / f"{method.name}_summary.json", summary)

    print(f"\nSummary for {method.name}")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
