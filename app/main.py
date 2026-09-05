from app.config.settings import settings


def main():
    print("Adaptive MAS")
    print("-" * 40)

    if settings.openrouter_api_key:
        print("OpenRouter API key: Loaded")
    else:
        print("OpenRouter API key: NOT FOUND")

    print(f"OpenRouter model: {settings.openrouter_model}")

    if settings.langchain_api_key:
        print("LangSmith API key: Loaded")
    else:
        print("LangSmith API key: NOT FOUND")

    print(f"LangSmith project: {settings.langchain_project}")


if __name__ == "__main__":
    main()
