# CyberGuardian Connect (Phase 1)

Connect a security web application (or upload a report), pull its incidents into
one common format, and preview them. The dashboard, AI analysis, copilot and
reports are built on top of this in later phases.

```
Source (web app / file)  ->  Connector  ->  Common incident format  ->  App
```

## Project layout

```
app.py                     Streamlit app: the Connections page
connectors/
  models.py                The common incident format (Incident and friends)
  base.py                  BaseConnector + ConnectorError
  rest_api.py              REST API connector (+ native -> common conversion)
  file_upload.py           File upload connector (text report -> common format)
source_app/
  data.py                  Sample incidents, simulate_new(), action log
  main.py                  FastAPI wrapper: the simulated SOC platform
sample_data/KnowledgeBase.txt   Example incident report for the file connector
tests/                     Unit tests (no web framework needed)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the demo (two terminals, both from this folder)

Terminal 1, the simulated web app:
```bash
uvicorn source_app.main:app --port 8000
```
Its API key is `demo-key-123` (change it with the `SOURCE_API_KEY` environment variable).
Interactive API docs: http://localhost:8000/docs

Terminal 2, CyberGuardian Connect:
```bash
streamlit run app.py
```

Then in the app:
1. Open **Add a connection**, choose **REST API**, keep the default URL and enter `demo-key-123`.
2. Press **Test connection**, then **Save and sync**. Four incidents appear.
3. Or choose **File upload** and upload `sample_data/KnowledgeBase.txt`.

## Live demo trick: create a new incident

While the app is open, create a new incident on the source and press **Sync**:

```bash
curl -X POST http://localhost:8000/api/v1/simulate -H "X-API-Key: demo-key-123"
```

(Phase 2 will refresh automatically, so the new incident appears on its own.)

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Adding another connector later

Create a class that extends `BaseConnector` with `test_connection()` and
`fetch_incidents()`, convert the source's data into `Incident`, and register it
in `connectors/__init__.py`. Nothing else in the app needs to change.

## Notes

- Connections live in the Streamlit session for now (they are lost on refresh).
  Phase 3 (accounts) moves them into a database with encrypted API keys.
- The API key is sent in the `X-API-Key` header. Use HTTPS for any real deployment.
