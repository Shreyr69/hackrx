# HackRX Intelligent Query Retrieval

## Setup

1. Python 3.10+
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment:

- `GEMINI_API_KEY` = your Google Generative Language API key

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API

POST `/hackrx/run`

Headers:
- `Content-Type: application/json`
- `Accept: application/json`
- `Authorization: Bearer 043dc79bbd910f6e4ea9b57b6705a94ee0677b8b3c80080823b643987dd73fe0`

Body:
```json
{
  "documents": "<PDF Blob URL>",
  "questions": ["..."]
}
```

Response:
```json
{ "answers": ["..."] }
```
