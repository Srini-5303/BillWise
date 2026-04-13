import re
import logging
from datetime import datetime
from google.cloud import vision

log    = logging.getLogger("ocr_pipeline")
client = vision.ImageAnnotatorClient()

def extract_text(path):
    with open(path, "rb") as img:
        content = img.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.text_annotations:
        return response.text_annotations[0].description
    return ""

def extract_date(text):
    text = text.replace("|", " ").replace(",", " ")
    text = re.sub(r"\s+", " ", text)

    patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{2}\b",
        r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\s+\d{1,2}\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\s+\d{4}\b",
        r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*/\d{1,2}/\d{4}\b",
    ]

    found = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            found.append(m.group())

    for raw in found:
        raw_clean = raw.strip()
        for fmt in [
            "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d",
            "%m-%d-%Y", "%m-%d-%y", "%b %d %Y",
            "%B %d %Y", "%d %b %Y", "%d %B %Y",
            "%b/%d/%Y", "%B/%d/%Y",
        ]:
            try:
                dt = datetime.strptime(raw_clean.title(), fmt)
                return dt.strftime("%Y-%m-%d")
            except:
                pass
    return ""

def extract_card_last4(text):
    lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
    card_keywords = ["VISA", "MASTERCARD", "CARD", "DEBIT", "CREDIT", "REFERENCE#"]

    for line in lines:
        if any(k in line for k in card_keywords):
            match = re.search(r"\d{4}", line)
            if match:
                return match.group()
        match = re.search(r"[X\*]{4,}\d{4}", line)
        if match:
            return match.group()[-4:]
    return "cash"

def detect_store(lines):
    for line in lines[:10]:
        clean = line.strip()
        if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", clean):
            continue
        if re.search(r"\d+\.\d{2}", clean):
            continue
        if len(clean) > 3:
            return clean
    return "UNKNOWN"


def extract_time(text):
    patterns = [
        r"\b\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)\b",
        r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b",
        r"\b\d{1,2}:\d{2}:\d{2}\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group().strip()
    return ""


def extract_subtotal(text):
    lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
    money_pattern = r"\d[\d,]*\.\d{2}"
    for line in lines:
        if "SUBTOTAL" in line or "SUB TOTAL" in line or "SUB-TOTAL" in line:
            m = re.search(money_pattern, line)
            if m:
                try:
                    return round(float(m.group().replace(",", "")), 2)
                except ValueError:
                    pass
    return ""


def extract_tax(text):
    lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
    money_pattern = r"\d[\d,]*\.\d{2}"
    for line in lines:
        if re.search(r"\bTAX\b", line) and "SUBTOTAL" not in line and "TOTAL" not in line:
            m = re.search(money_pattern, line)
            if m:
                try:
                    return round(float(m.group().replace(",", "")), 2)
                except ValueError:
                    pass
    return ""


def extract_payment_method(text):
    upper = text.upper()
    for keyword in ["VISA", "MASTERCARD", "MASTER CARD", "AMEX",
                    "AMERICAN EXPRESS", "DISCOVER", "DEBIT", "CREDIT", "CASH"]:
        if keyword in upper:
            return keyword.replace(" ", "_")
    return ""


def extract_receipt_number(text):
    patterns = [
        r"(?:RECEIPT|TRANS(?:ACTION)?|REF(?:ERENCE)?|ORDER|CHECK|TICKET)\s*[#:No.]*\s*([A-Z0-9\-]{4,})",
        r"(?:RECEIPT\s*(?:NO|NUMBER|#))\s*[:\-]?\s*([A-Z0-9\-]{4,})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def extract_items(text):
    """
    Extract purchased line items from receipt text.
    Looks for lines that have a description followed by a price,
    skips totals, taxes, subtotals, and other non-item lines.
    Returns a list of (item_name, item_price) tuples.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    skip_keywords = [
        "TOTAL", "SUBTOTAL", "TAX", "BALANCE", "CHANGE", "CASH",
        "CREDIT", "DEBIT", "VISA", "MASTERCARD", "CARD", "DUE",
        "AMOUNT", "PAYMENT", "THANK", "SAVE", "DISCOUNT", "COUPON",
        "MEMBER", "REWARD", "POINT", "RECEIPT", "STORE", "TEL",
        "ADDRESS", "PHONE", "WWW", "HTTP", "APPROVED", "AUTH"
    ]

    money_pattern = r"\d[\d,]*\.\d{2}"
    items = []

    for line in lines:
        upper = line.upper()

        # Must contain a price to be a line item
        price_match = re.search(money_pattern, line)
        if not price_match:
            continue

        # Skip known non-item lines
        if any(k in upper for k in skip_keywords):
            continue

        # Skip lines that are only a number/price
        name = re.sub(money_pattern, "", line).strip(" .-@#*/\\")
        if len(name) < 2:
            continue

        items.append((name, price_match.group(0)))

    return items if items else []


def extract_total_from_text(text):
    lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
    money_pattern = r"\d[\d,]*\.\d{2}"
    amounts = []
    keyword_amounts = []

    for line in lines:
        clean_line = line.replace(",", "")
        matches = re.findall(money_pattern, clean_line)
        for m in matches:
            value = float(m)
            if 0 < value < 20000:
                amounts.append(value)
                if any(k in line for k in ["TOTAL", "AMOUNT", "BALANCE", "DUE", "Purchase"]):
                    keyword_amounts.append(value)

    if keyword_amounts:
        return max(keyword_amounts)
    if amounts:
        return max(amounts)
    return ""

def process_image(path):
    """
    Master function — runs full pipeline on a single image.
    Returns a dict of extracted fields.
    """
    log.info("Calling Google Cloud Vision OCR on %s", path)
    text  = extract_text(path)
    log.info("OCR returned %d characters", len(text))

    lines          = text.split("\n")
    store          = detect_store(lines)
    date           = extract_date(text)
    time           = extract_time(text)
    subtotal       = extract_subtotal(text)
    tax            = extract_tax(text)
    total          = extract_total_from_text(text)
    card           = extract_card_last4(text)
    payment_method = extract_payment_method(text)
    receipt_number = extract_receipt_number(text)
    items          = extract_items(text)

    log.info("Extracted — store=%r | date=%r | total=%r | card=%r | items=%d",
             store, date, total, card, len(items))

    return {
        "store"          : store,
        "date"           : date,
        "time"           : time,
        "subtotal"       : subtotal,
        "tax"            : tax,
        "total"          : total,
        "card"           : card,
        "payment_method" : payment_method,
        "receipt_number" : receipt_number,
        "items"          : items,
    }
