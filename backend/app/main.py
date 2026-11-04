from fastapi import FastAPI

from backend.app.api.routes.analysis import router as analysis_router
from backend.app.api.routes.csv_read import router as csv_read_router
from backend.app.api.routes.data_quality import router as data_quality_router
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


app.include_router(analysis_router)
app.include_router(upload_router)
app.include_router(csv_read_router)
app.include_router(profile_csv_router)
app.include_router(metadata_router)
app.include_router(data_quality_router)
