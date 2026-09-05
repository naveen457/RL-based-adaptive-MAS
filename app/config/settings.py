import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")

    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )

    openrouter_model: str = os.getenv(
        "OPENROUTER_MODEL",
        "",
    )

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


settings = Settings()
