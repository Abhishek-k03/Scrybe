from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "WebOracle"
    VERSION: str = "0.1.0"

    GROQ_API_KEY: str = ""
    JINA_API_KEY: str = ""

    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    CHROMA_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "weboracle"


settings = Settings()
