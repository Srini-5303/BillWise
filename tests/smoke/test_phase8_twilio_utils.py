from apps.twilio_ingestion.app import create_app


def test_phase8_twilio_app_creation():
    app = create_app()
    client = app.test_client()

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json["status"] == "ok"