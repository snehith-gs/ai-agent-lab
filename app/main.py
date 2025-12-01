from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging import configure_logging
from app.core.settings import settings
from app.routers import health, chat, tools, docs

configure_logging()
app = FastAPI(title=settings.app_name, 
              version=settings.version,
              description="Backend API for building intelligent AI agents.")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# CORS (open during dev; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(tools.router)
app.include_router(docs.router) 
