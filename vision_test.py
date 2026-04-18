import sys
from google.cloud import vision
from ocr_pipeline import extract_text, process_image


def main():
    if len(sys.argv) < 2:
        print("Usage: python vision_test.py <path_to_image>")
        sys.exit(1)

    path = sys.argv[1]

    print("\n==============================")
    print("FILE:", path)
    print("==============================\n")

    raw = extract_text(path)
    print("── RAW OCR TEXT ──────────────────────────")
    print(raw)
    print("──────────────────────────────────────────\n")

    result = process_image(path)

    print(f"Store          : {result['store']}")
    print(f"Date           : {result['date']}")
    print(f"Time           : {result['time']}")
    print(f"Subtotal       : {result['subtotal']}")
    print(f"Tax            : {result['tax']}")
    print(f"Total          : {result['total']}")
    print(f"Card           : {result['card']}")
    print(f"Payment Method : {result['payment_method']}")
    print(f"Receipt No     : {result['receipt_number']}")
    print(f"\nItems ({len(result['items'])}):")
    for name, price in result["items"]:
        print(f"  {price:>10}  {name}")

    print("\n==============================")
    print("END OCR")
    print("==============================\n")


if __name__ == "__main__":
    main()