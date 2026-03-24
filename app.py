import os
import hashlib
import tempfile
import requests
from flask import Flask, request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from ocr_pipeline import process_image
from csv_writer import append_bill, is_duplicate

app = Flask(__name__)

ACCOUNT_SID   = os.environ["TWILIO_ACCOUNT_SID"]
AUTH_TOKEN    = os.environ["TWILIO_AUTH_TOKEN"]
twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


def download_image(media_url: str) -> tuple[str, str]:
    """
    Download image from Twilio. Returns (temp_file_path, md5_hash).
    Hash is computed on raw bytes before writing to disk.
    """
    resp = requests.get(media_url, auth=(ACCOUNT_SID, AUTH_TOKEN), timeout=30)
    resp.raise_for_status()

    image_bytes  = resp.content
    image_hash   = hashlib.md5(image_bytes).hexdigest()

    content_type = resp.headers.get("Content-Type", "image/jpeg")
    ext          = "." + content_type.split("/")[-1].split(";")[0].strip()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(image_bytes)
    tmp.close()

    return tmp.name, image_hash


@app.route("/webhook", methods=["POST"])
def webhook():
    num_media = int(request.form.get("NumMedia", 0))
    sender    = request.form.get("From", "unknown")
    resp      = MessagingResponse()

    if num_media == 0:
        resp.message("👋 Hi! Send me a bill image and I'll extract the details 🧾")
        return str(resp)

    reply_parts = []

    for i in range(num_media):
        media_url  = request.form.get(f"MediaUrl{i}")
        media_type = request.form.get(f"MediaContentType{i}", "")

        if media_type not in SUPPORTED_TYPES:
            reply_parts.append(f"⚠️ Attachment {i+1}: unsupported type ({media_type}). Send JPG or PNG.")
            continue

        img_path = None
        try:
            img_path, image_hash = download_image(media_url)

            # ── Layer 1: Check image hash before running OCR ──
            dupe, match = is_duplicate(image_hash, "", "", "")
            if dupe:
                reply_parts.append(
                    f"⚠️ This bill has already been submitted.\n"
                    f"🧾 Original entry: Bill #{match[0]} | {match[2]} | {match[3]} | ${match[4]}"
                )
                continue

            # ── Run OCR ──
            result = process_image(img_path)

            # ── Layer 2: Check extracted fields after OCR ──
            dupe, match = is_duplicate(
                image_hash,
                result["store"],
                result["date"],
                result["total"]
            )
            if dupe:
                reply_parts.append(
                    f"⚠️ This bill has already been submitted.\n"
                    f"🧾 Original entry: Bill #{match[0]} | {match[2]} | {match[3]} | ${match[4]}"
                )
                continue

            # ── Save to CSV ──
            serial = append_bill(
                filename   = os.path.basename(img_path),
                store      = result["store"],
                date       = result["date"],
                total      = result["total"],
                card       = result["card"],
                sender     = sender,
                image_hash = image_hash,
            )

            reply_parts.append(
                f"✅ *Bill #{serial} Saved*\n"
                f"🏪 Store  : {result['store']}\n"
                f"📅 Date   : {result['date']  or 'Not found'}\n"
                f"💳 Card   : {result['card']}\n"
                f"💰 Total  : {result['total'] or 'Not found'}"
            )

        except Exception as e:
            reply_parts.append(f"❌ Could not process image {i+1}: {str(e)}")

        finally:
            if img_path and os.path.exists(img_path):
                os.unlink(img_path)

    resp.message("\n\n".join(reply_parts))
    return str(resp)


@app.route("/", methods=["GET"])
def health():
    return "Bill Bot is running ✅", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)