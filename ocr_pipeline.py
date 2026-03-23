import re
from datetime import datetime
from google.cloud import vision

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
                if any(k in line for k in ["TOTAL", "AMOUNT", "BALANCE", "DUE"]):
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
    text  = extract_text(path)
    lines = text.split("\n")
    return {
        "store" : detect_store(lines),
        "date"  : extract_date(text),
        "total" : extract_total_from_text(text),
        "card"  : extract_card_last4(text),
    }