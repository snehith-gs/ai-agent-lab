from fastapi import FastAPI
from app.routers import health, chat

app = FastAPI(
    title="AI Agent Lab",
    version="0.1.0",
    description="Backend API for building intelligent AI agents."
)

# Register routers
app.include_router(health.router)
app.include_router(chat.router)
