# AI Agent Lab – Backend MVP

This project is the backend for your personal AI Agent system using FastAPI.

## Architecture
                    ┌────────────────────────────────────┐
                    │            Frontend (later)        │
                    │  React/Next.js chat UI             │
                    │  - sends /chat, /ask requests      │
                    └───────────────▲────────────────────┘
                                    │ HTTP (JSON)
                                    │
                        ┌───────────┴───────────┐
                        │  FastAPI Backend      │
                        │  (ai-agent-lab)       │
                        │                       │
             ┌──────────┴──────────┐     ┌─────┴────────────────┐
             │  Chat Router        │     │  Retrieval Router     │
             │  /chat              │     │  /ingest, /ask        │
             └──────────▲──────────┘     └───────────▲──────────┘
                        │                            │
                        │ calls                      │ calls
                        │                            │
               ┌────────┴───────────┐        ┌───────┴─────────────────────┐
               │  LLM Service       │        │  RAG Service                 │
               │  (OpenAI client)   │        │  - chunk & embed documents   │
               │                    │        │  - upsert to Vector DB       │
               │  - handles models  │        │  - semantic search (top-k)   │
               │  - API errors      │        │  - build augmented prompt     │
               └────────▲───────────┘        └─────────▲────────────────────┘
                        │                               │
                        │ OpenAI API                    │ Qdrant (Vector DB)
                        │                               
              ┌─────────┴──────────┐          ┌─────────┴──────────┐
              │ Provider (LLM)     │          │ Qdrant (Docker)    │
              │ gpt-4o-mini, etc.  │          │ embeddings index    │
              └────────────────────┘          └─────────────────────┘


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
