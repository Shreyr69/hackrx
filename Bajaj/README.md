# HackRX Intelligent Query Retrieval

## Local Setup

1. Python 3.10+
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment:

- `OPENAI_API_KEY` = your OpenAI API key

## Local Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Deploy to Render

1. **Push your code to GitHub** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Sign up/Login
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure the service**:
   - **Name**: `hackrx-api` (or any name you prefer)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

4. **Set Environment Variables**:
   - Go to your service dashboard
   - Click "Environment" tab
   - Add environment variable:
     - **Key**: `OPENAI_API_KEY`
     - **Value**: Your actual OpenAI API key

5. **Deploy**:
   - Click "Create Web Service"
   - Render will automatically build and deploy your app

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

## Test Deployed API

Once deployed, your API will be available at:
`https://your-app-name.onrender.com/hackrx/run`

Test with PowerShell:
```powershell
$headers = @{
  "Content-Type"="application/json"
  "Accept"="application/json"
  "Authorization"="Bearer 043dc79bbd910f6e4ea9b57b6705a94ee0677b8b3c80080823b643987dd73fe0"
}
$body = @{
  documents = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
  questions = @(
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?"
  )
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri "https://your-app-name.onrender.com/hackrx/run" -Headers $headers -Body $body
```
