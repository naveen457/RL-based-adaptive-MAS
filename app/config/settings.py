import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    # NVIDIA NIM Configuration
    api_key: str = os.getenv("NVIDIA_API_KEY", "")
    base_url: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model: str = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4096"))

    # MongoDB Atlas Configuration
    mongodb_uri: str = os.getenv("MONGODB_URI", "")
    mongodb_db_name: str = os.getenv("MONGODB_DB_NAME", "adaptive_mas")

    langchain_tracing_v2: bool = (
        os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    )

    langchain_endpoint: str = os.getenv(
        "LANGCHAIN_ENDPOINT",
        "https://api.smith.langchain.com",
    )

    langchain_api_key: str = os.getenv(
        "LANGCHAIN_API_KEY",
        "",
    )

    langchain_project: str = os.getenv(
        "LANGCHAIN_PROJECT",
        "adaptive-mas",
    )

    tavily_api_key: str = os.getenv(
        "TAVILY_API_KEY",
        "",
    )

    # Critic Quality Threshold & Feedback Looping
    critic_quality_threshold: float = float(os.getenv("CRITIC_QUALITY_THRESHOLD", "0.75"))
    critic_max_retries: int = int(os.getenv("CRITIC_MAX_RETRIES", "1"))


settings = Settings()

