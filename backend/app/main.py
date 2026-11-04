from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes.agent import router as agent_router
from backend.app.api.routes.analysis import router as analysis_router
from backend.app.api.routes.csv_read import router as csv_read_router
from backend.app.api.routes.data_quality import router as data_quality_router
from backend.app.api.routes.database_analysis import router as database_analysis_router
from backend.app.api.routes.database_ingestion import router as database_ingestion_router
from backend.app.api.routes.documents import router as documents_router
from backend.app.api.routes.insights import router as insights_router
from backend.app.api.routes.llm import router as llm_router
from backend.app.api.routes.metadata import router as metadata_router
from backend.app.api.routes.profile_csv import router as profile_csv_router
from backend.app.api.routes.upload import router as upload_router
from backend.app.core.config import settings
from backend.app.db.ingestion import DatabaseIngestionService
from backend.app.ingestion.upload_service import UploadService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Upload records first: ingestion entries are only reloaded for uploads that still exist.
    UploadService.load_registry()
    DatabaseIngestionService.load_registry()
    yield


FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
SAMPLE_DATA_DIR = Path(__file__).resolve().parents[2] / "sample_data"

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root(request: Request):
    # Browsers land on the web app; API clients keep getting the JSON status payload.
    if FRONTEND_DIR.is_dir() and "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url="/app/")
    return {"message": "AI Data Analyst Agent API is running.", "app_name": settings.APP_NAME, "app_env": settings.APP_ENV, "debug": settings.DEBUG}


@app.get("/health")
def health_check():
    return {"status": "healthy", "app_env": settings.APP_ENV, "debug": settings.DEBUG}


app.include_router(agent_router)
app.include_router(analysis_router)
app.include_router(upload_router)
app.include_router(documents_router)
app.include_router(csv_read_router)
app.include_router(profile_csv_router)
app.include_router(metadata_router)
app.include_router(data_quality_router)
app.include_router(database_ingestion_router)
app.include_router(database_analysis_router)
app.include_router(llm_router)
app.include_router(insights_router)

if FRONTEND_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
if SAMPLE_DATA_DIR.is_dir():
    app.mount("/sample-data", StaticFiles(directory=SAMPLE_DATA_DIR), name="sample-data")
