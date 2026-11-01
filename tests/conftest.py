import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

# Ensure mock credentials exist for CI and local unit tests when no .env is present
if not os.environ.get("NVIDIA_API_KEY"):
    os.environ["NVIDIA_API_KEY"] = "nvapi-mock-ci-key-for-testing"
if not os.environ.get("NVIDIA_MODEL"):
    os.environ["NVIDIA_MODEL"] = "nvidia/nemotron-3.5-lightning-30b-a3b"

from app.config.settings import settings

if not settings.api_key:
    settings.api_key = "nvapi-mock-ci-key-for-testing"
if not settings.model:
    settings.model = "nvidia/nemotron-3.5-lightning-30b-a3b"
