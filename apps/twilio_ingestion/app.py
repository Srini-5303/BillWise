from __future__ import annotations

import os
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, Response, request
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse

from billwise.common.logging import get_logger
from billwise.dashboard.exports import export_dashboard_views
from billwise.pipeline.orchestrator import run_billwise_pipeline


def create_app() -> Flask:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=True)

    app = Flask(__name__)
    logger = get_logger("billwise.twilio")

    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    verify_signature = (os.getenv("TWILIO_VERIFY_SIGNATURE", "false").strip().lower() == "true")
    auto_export_views = (os.getenv("TWILIO_AUTO_EXPORT_VIEWS", "true").strip().lower() == "true")

    validator = RequestValidator(twilio_token) if twilio_token else None

    def _validate_request() -> bool:
        if not verify_signature:
            return True
        if validator is None:
            return False

        signature = request.headers.get("X-Twilio-Signature", "")
        url = request.url
        form = request.form.to_dict(flat=True)
        return validator.validate(url, form, signature)

    def _download_twilio_media(media_url: str, suffix: str = ".jpg") -> Path:
        if not twilio_sid or not twilio_token:
            raise RuntimeError("Twilio credentials not configured")

        resp = requests.get(media_url, auth=(twilio_sid, twilio_token), timeout=60)
        resp.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(resp.content)
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok"}, 200

    @app.route("/twilio/webhook", methods=["POST"])
    def twilio_webhook():
        if not _validate_request():
            return Response("Invalid Twilio signature", status=403)

        num_media = int(request.form.get("NumMedia", "0") or "0")
        twiml = MessagingResponse()
        msg = twiml.message()

        if num_media == 0:
            msg.body("Please send a receipt image to BillWise.")
            return Response(str(twiml), mimetype="application/xml")

        media_url = request.form.get("MediaUrl0", "")
        media_type = request.form.get("MediaContentType0", "image/jpeg")

        if not media_type.startswith("image/"):
            msg.body("Please send an image receipt. PDF support can be added later.")
            return Response(str(twiml), mimetype="application/xml")

        suffix = ".jpg"
        if "png" in media_type:
            suffix = ".png"
        elif "webp" in media_type:
            suffix = ".webp"

        temp_path = None

        try:
            logger.info("Downloading media from Twilio: %s", media_type)
            temp_path = _download_twilio_media(media_url, suffix=suffix)

            result = run_billwise_pipeline(temp_path)

            export_note = ""
            if auto_export_views:
                try:
                    export_result = export_dashboard_views()
                    logger.info(
                        "Twilio-triggered dashboard export completed | receipts=%s items=%s joined=%s",
                        export_result["receipts_rows"],
                        export_result["items_rows"],
                        export_result["joined_rows"],
                    )
                    export_note = "\nDashboard refreshed: Yes"
                except Exception:
                    logger.exception("Dashboard export failed after Twilio pipeline run")
                    export_note = "\nDashboard refreshed: No"

            msg.body(
                f"BillWise processed your receipt.\n"
                f"Vendor: {result.vendor_name or 'Unknown'}\n"
                f"Items: {result.item_count}\n"
                f"Review needed: {'Yes' if result.requires_review else 'No'}\n"
                f"Receipt ID: {result.receipt_id}"
                f"{export_note}"
            )

        except Exception as e:
            logger.exception("Twilio receipt processing failed")
            msg.body(f"BillWise could not process the receipt. Error: {str(e)}")

        finally:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass

        return Response(str(twiml), mimetype="application/xml")

    return app