from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database.database import Base, engine
from app.database import models
from sqlalchemy import inspect, text
from app.routes import upload, analysis, reports
Base.metadata.create_all(bind=engine)
if "analyses" in inspect(engine).get_table_names() and "audit_json" not in {column["name"] for column in inspect(engine).get_columns("analyses")}:
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE analyses ADD COLUMN audit_json TEXT"))
app = FastAPI(title="SemanticGuard AI", version="1.0.0", description="Human-review semantic similarity analysis for student assignments")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(upload.router); app.include_router(analysis.router); app.include_router(reports.router)
@app.get("/api/health")
def health(): return {"status": "ok", "model": settings.model_name}
