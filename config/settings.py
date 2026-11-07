"""
Service 01: Configuration and Settings

Provides centralized configuration for the RFP Intelligence Platform.
Reads values from .env file with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_LLM_KEY = os.getenv("NVIDIA_LLM_KEY")
NVIDIA_EMBED_KEY = os.getenv("NVIDIA_EMBED_KEY")
NVIDIA_RERANK_KEY = os.getenv("NVIDIA_RERANK_KEY")

# Endpoints
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
RERANK_URL = os.getenv("RERANK_URL", "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking")

# Model configuration
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3-embed-1b")
RERANK_MODEL = os.getenv("RERANK_MODEL", "nvidia/nemotron-3.5-rerank-8b")

# Rate limits and performance
MAX_RPM = int(os.getenv("MAX_RPM", "30"))
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "3"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "20"))
MAX_LLM_CALLS_PER_RUN = int(os.getenv("MAX_LLM_CALLS_PER_RUN", "150"))

# Chunking parameters
CHUNK_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "450"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "700"))
CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "80"))
CHUNK_OVERLAP_RATIO = float(os.getenv("CHUNK_OVERLAP_RATIO", "0.15"))

# Search settings
DENSE_TOP_N = int(os.getenv("DENSE_TOP_N", "30"))
BM25_TOP_N = int(os.getenv("BM25_TOP_N", "30"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))

# Retry and timeout
MAX_RETRIES_PER_FIELD = int(os.getenv("MAX_RETRIES_PER_FIELD", "2"))
EVIDENCE_CHUNKS_PER_FIELD = int(os.getenv("EVIDENCE_CHUNKS_PER_FIELD", "4"))
NODE_TIMEOUT_S = int(os.getenv("NODE_TIMEOUT_S", "60"))

# Q&A settings
QA_TOP_K = int(os.getenv("QA_TOP_K", "6"))
QA_VERIFY_ENABLED = os.getenv("QA_VERIFY_ENABLED", "true").lower() == "true"
SUMMARY_MIN_SENTENCES = int(os.getenv("SUMMARY_MIN_SENTENCES", "3"))
SUMMARY_MAX_SENTENCES = int(os.getenv("SUMMARY_MAX_SENTENCES", "6"))

# Key fallback: nvidia_api_key falls back to nvidia_llm_key if not set
# Per design: at least one key must exist
def get_llm_key():
    """Get LLM key, falling back to NVIDIA_API_KEY."""
    key = NVIDIA_LLM_KEY or NVIDIA_API_KEY
    if not key:
        raise ValueError("No LLM key configured. Set NVIDIA_LLM_KEY or NVIDIA_API_KEY in .env")
    return key

def get_embed_key():
    """Get embed key, falling back to NVIDIA_API_KEY."""
    key = NVIDIA_EMBED_KEY or NVIDIA_API_KEY
    if not key:
        raise ValueError("No embed key configured. Set NVIDIA_EMBED_KEY or NVIDIA_API_KEY in .env")
    return key

def get_rerank_key():
    """Get rerank key, falling back to NVIDIA_API_KEY."""
    key = NVIDIA_RERANK_KEY or NVIDIA_API_KEY
    if not key:
        raise ValueError("No rerank key configured. Set NVIDIA_RERANK_KEY or NVIDIA_API_KEY in .env")
    return key


# Optional per-agent model overrides
LLM_MODEL_EXTRACT = os.getenv("LLM_MODEL_EXTRACT", "")
LLM_MODEL_VALIDATE = os.getenv("LLM_MODEL_VALIDATE", "")
LLM_MODEL_RECONCILE = os.getenv("LLM_MODEL_RECONCILE", "")
LLM_MODEL_QA = os.getenv("LLM_MODEL_QA", "")


# Convenience: use override if set, otherwise default model
def get_model(model_type: str) -> str:
    """Get model name, respecting per-agent overrides."""
    override_map = {
        "extract": LLM_MODEL_EXTRACT or "",
        "validate": LLM_MODEL_VALIDATE or "",
        "reconcile": LLM_MODEL_RECONCILE or "",
        "qa": LLM_MODEL_QA or "",
    }
    override = override_map.get(model_type, "")
    if override:
        return override
    return globals()[f"LLM_MODEL" if model_type != "embed" else "EMBED_MODEL" if model_type != "rerank" else "RERANK_MODEL"]


# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
PARSSED_DIR = STORAGE_DIR / "parsed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TRACES_DIR = OUTPUTS_DIR / "traces"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for d in [DATA_DIR, STORAGE_DIR, PARSSED_DIR, OUTPUTS_DIR, TRACES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)