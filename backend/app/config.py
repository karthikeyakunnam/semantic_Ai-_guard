from pathlib import Path
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[1]
class Settings(BaseSettings):
    database_url: str = f"sqlite:///{ROOT / 'semanticguard.db'}"
    upload_dir: Path = ROOT / "uploads"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_upload_mb: int = 20
    min_similarity: float = 0.45
    top_k: int = 5
    boilerplate_phrases: list[str] = []
    cors_origins: list[str] = ["http://localhost:5173"]
settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
