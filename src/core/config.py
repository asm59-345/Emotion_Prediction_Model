from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Moodline Pro API"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # Model & Tokenizer Paths
    MODEL_PATH: str = "Artifacts/BiGRU_Modle.keras"
    TOKENIZER_PATH: str = "Artifacts/tokenizer.pkl"
    MAX_SEQUENCE_LENGTH: int = 50
    EMBEDDING_DIM: int = 300
    VOCAB_SIZE: int = 10000
    
    # Database Settings (SQLite SQL storage)
    DATABASE_URL: str = "sqlite:///./data/moodline_analytics.db"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 120
    
    # Emotion Labels & Emojis
    EMOTION_LABELS: List[str] = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    EMOTION_EMOJIS: dict = {
        "sadness": "😢",
        "joy": "😊",
        "love": "❤️",
        "anger": "😡",
        "fear": "😨",
        "surprise": "😲"
    }

settings = Settings()
