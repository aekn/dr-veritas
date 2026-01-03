from pydantic import BaseModel, PostgresDsn, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    """Database connectivity settings."""

    pg_dsn: PostgresDsn = Field(...)


class BlobSettings(BaseModel):
    """Blob storage settings.

    Using local fs for prototyping, planning to move to gcs as
    knowledge base grows.
    """

    base_uri: str = Field(...)
    prefix: str = "vera"


class ChunkingSettings(BaseModel):
    """Document chunking settings."""

    version: str = "v1"
    max_tokens: int = 900
    overlap_tokens: int = 120


class JatsSettings(BaseModel):
    """Extraction and quality settings for JATS XML parsing."""

    version: str = "v1"
    # bad qa metrics?? as xml document tree is messy
    # min_words: int = 1000
    # min_sections: int = 4
    # min_paragraphs: int = 20


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="VERA_",
        env_nested_delimiter="__",
    )

    env: str = "local"
    db: DatabaseSettings
    blob: BlobSettings
    chunking: ChunkingSettings = ChunkingSettings()
    jats: JatsSettings = JatsSettings()


settings = Settings()
