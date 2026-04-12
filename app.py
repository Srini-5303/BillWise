import os
import hashlib
import tempfile
import requests
from flask import Flask, request, render_template, jsonify
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from ocr_pipeline import process_image
from csv_writer import append_bill, is_duplicate
from chatbot import handle_chat_message, reload_session, clear_session
import categorizer

app = Flask(__name__)

# Load DistilBERT model once at startup
categorizer.init()

ACCOUNT_SID   = os.environ["TWILIO_ACCOUNT_SID"]
AUTH_TOKEN    = os.environ["TWILIO_AUTH_TOKEN"]
twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}

# Keywords that identify a text message as a spending question
_QUESTION_STARTERS = {
    "how", "what", "when", "show", "list", "which", "total",
    "sum", "average", "who", "where", "count", "give", "find",
    "reset", "clear", "reload", "refresh",
}


def _is_question(text: str) -> bool:
    """Return True if the SMS text looks like a chatbot query."""
    if not text:
        return False
    lower = text.lower().strip()
    first_word = lower.split()[0] if lower else ""
    return lower.endswith("?") or first_word in _QUESTION_STARTERS


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


# ── Twilio webhook ─────────────────────────────────────────────────────────────

@app.route("/webhook", methods=["POST"])
def webhook():
    num_media = int(request.form.get("NumMedia", 0))
    sender    = request.form.get("From", "unknown")
    resp      = MessagingResponse()

    # ── Text-only message → route to chatbot ──────────────────────────────────
    if num_media == 0:
        text = request.form.get("Body", "").strip()
        if _is_question(text):
            answer = handle_chat_message(session_id=sender, message=text)
            resp.message(answer)
        else:
            resp.message(
                "👋 Hi! Send me a bill image to digitize it 🧾\n"
                "Or ask a question about your spending — e.g. "
                '"How much did we spend last month?"'
            )
        return str(resp)

    # ── Image message → OCR pipeline ─────────────────────────────────────────
    reply_parts = []

    for i in range(num_media):
        media_url  = request.form.get(f"MediaUrl{i}")
        media_type = request.form.get(f"MediaContentType{i}", "")

        if media_type not in SUPPORTED_TYPES:
            reply_parts.append(
                f"⚠️ Attachment {i+1}: unsupported type ({media_type}). Send JPG or PNG."
            )
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

            # ── Categorize each item ──
            categorized_items = [
                (name, price, categorizer.categorize(name))
                for name, price in result["items"]
            ]

            # ── Save to CSV ──
            serial = append_bill(
                filename   = os.path.basename(img_path),
                store      = result["store"],
                date       = result["date"],
                total      = result["total"],
                card       = result["card"],
                sender     = sender,
                image_hash = image_hash,
                items      = categorized_items,
            )

            # Reload cached CSV for this sender so next query sees new bill
            reload_session(sender)

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


# ── Web chat UI ────────────────────────────────────────────────────────────────

@app.route("/chat")
def chat_ui():
    return render_template("chat.html")


# ── REST API ───────────────────────────────────────────────────────────────────

@app.route("/api/query", methods=["POST"])
def api_query():
    """
    JSON body: { "session_id": str, "question": str, "csv_source": str (optional) }
    Returns  : { "answer": str, "success": bool }
    """
    data       = request.get_json(force=True) or {}
    session_id = data.get("session_id", "web-default")
    question   = data.get("question", "").strip()
    csv_source = data.get("csv_source")

    if not question:
        return jsonify({"success": False, "error": "No question provided."}), 400

    try:
        answer = handle_chat_message(
            session_id=session_id,
            message=question,
            csv_source=csv_source,
        )
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Clear conversation history for a session."""
    data       = request.get_json(force=True) or {}
    session_id = data.get("session_id", "web-default")
    clear_session(session_id)
    return jsonify({"success": True, "message": "Session cleared."})


@app.route("/api/reload", methods=["POST"])
def api_reload():
    """Force a CSV reload for a session (e.g. after new bills are added)."""
    data       = request.get_json(force=True) or {}
    session_id = data.get("session_id", "web-default")
    reload_session(session_id)
    return jsonify({"success": True, "message": "Data will be reloaded on next query."})


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return "Bill Bot is running ✅", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
