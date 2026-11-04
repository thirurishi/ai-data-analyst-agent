from fastapi import FastAPI

from backend.app.api.routes.agent import router as agent_router
from backend.app.api.routes.analysis import router as analysis_router
from backend.app.api.routes.csv_read import router as csv_read_router
from backend.app.api.routes.data_quality import router as data_quality_router
from backend.app.api.routes.database_analysis import router as database_analysis_router
from backend.app.api.routes.database_ingestion import router as database_ingestion_router
from backend.app.api.routes.documents import router as documents_router
from backend.app.api.routes.llm import router as llm_router
from backend.app.api.routes.metadata import router as metadata_router
from backend.app.api.routes.profile_csv import router as profile_csv_router
from backend.app.api.routes.upload import router as upload_router
from backend.app.core.config import settings

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)


@app.get("/")
def read_root():
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
