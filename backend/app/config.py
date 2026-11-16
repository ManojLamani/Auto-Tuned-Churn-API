import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
