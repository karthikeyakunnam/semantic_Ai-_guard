from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.config import settings
@lru_cache(maxsize=1)
def get_model(): return SentenceTransformer(settings.model_name)
def encode(texts: list[str]):
    return get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
