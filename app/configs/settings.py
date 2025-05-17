import os

import dotenv
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class Settings(BaseSettings):
    API_PREFIX: str = "/api/v1"
    SERVER_ADDRESS: str = os.environ.get("SERVER_ADDRESS", "0.0.0.0")
    SERVER_PORT: int = int(os.environ.get("SERVER_PORT", 80))
    SERVER_WORKERS: int = int(os.environ.get("SERVER_WORKERS", 1))
    SERVER_URL: str = f"{SERVER_ADDRESS}:{SERVER_PORT}{API_PREFIX}"

    # DB Connection Info
    DB_USER: str = os.environ.get("DB_USER")
    DB_PASSWORD: str = os.environ.get("DB_PASSWORD")
    DB_ADDRESS: str = os.environ.get("DB_ADDRESS")
    DB_PORT: int = int(os.environ.get("DB_PORT"))
    DB_NAME: str = os.environ.get("DB_NAME")
    DATABASE_URL: str = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_ADDRESS}:{DB_PORT}/{DB_NAME}"

    # Zenko Storage Info
    ZENKO_SERVER_URL: str = os.environ.get("ZENKO_SERVER_URL")
    MODEL_BUCKET_NAME: str = os.environ.get("MODEL_BUCKET_NAME", "model")
    EVALUATION_BUCKET_NAME: str = os.environ.get("EVALUATION_BUCKET_NAME", "evaluation")
    SCALITY_ACCESS_KEY_ID: str = os.environ.get("SCALITY_ACCESS_KEY_ID")
    SCALITY_SECRET_ACCESS_KEY: str = os.environ.get("SCALITY_SECRET_ACCESS_KEY")


settings = Settings()
