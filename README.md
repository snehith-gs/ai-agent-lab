# AI Agent Lab – Backend MVP

This project is the backend for your personal AI Agent system using FastAPI.

## ✅ Features
- FastAPI backend
- `/health` endpoint
- `/chat` endpoint using OpenAI LLMs
- Clean project structure
- Ready for vector DB + agent upgrades

## ✅ Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="your-key-here"
uvicorn app.main:app --reload
