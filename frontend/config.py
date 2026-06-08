import os
from dotenv import load_dotenv

# Load .env once for the whole frontend
load_dotenv()

# Single source of truth for the API host.
# Each page composes its own endpoint path from this base.
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Endpoint paths, derived from the base
PREDICT_URL = f"{API_BASE_URL}/predict-heart-disease"
DATASET_STATS_URL = f"{API_BASE_URL}/dataset-stats"
MODEL_INFO_URL = f"{API_BASE_URL}/model-info"
HEALTH_URL = f"{API_BASE_URL}/health"

# Shared page metadata
APP_TITLE = "MedBuddy.ML"
APP_ICON = "⚕️"