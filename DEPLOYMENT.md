# BillWise Deployment Notes

---

## Local Development

### Start dashboard

```powershell
python apps\dashboard\run_dashboard.py
```

### Start Twilio ingestion service

```powershell
python scripts\run_twilio_ingestion.py
```

### Start ngrok for Twilio local testing

```powershell
ngrok http 8080
```

Then set the following in your Twilio Sandbox:

| Field | Value |
|---|---|
| When a Message Comes In | `https://<your-ngrok-url>/twilio/webhook` |
| Method | `POST` |

---

## Production Considerations

### Dashboard

Recommended production options:

- Streamlit deployment behind a reverse proxy
- Containerized deployment
- Managed VM or container platform

### Twilio Ingestion

The current Flask app is suitable for development. For production, use:

- `gunicorn` or another production WSGI server
- A real public HTTPS endpoint
- `TWILIO_VERIFY_SIGNATURE=true`

### Secrets

Store secrets outside git using one of:

- Environment variables
- A secret manager
- Mounted credential files

### GCS

BillWise currently supports hybrid storage:

- Local SQLite and local artifacts
- Mirrored GCS artifacts and exports

---

## Production Checklist

- [ ] Set `TWILIO_VERIFY_SIGNATURE=true`
- [ ] Use a real public domain instead of ngrok
- [ ] Move credentials outside the repo
- [ ] Keep `.env` out of version control
- [ ] Rotate service account credentials if ever exposed locally