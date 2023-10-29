import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MANATAL_ENV: str = os.environ.get("MANATAL_ENV") or "development"
    LAYOUT_BACKEND: str


@lru_cache()
def get_settings() -> Settings:
    environment = os.environ.get("MANATAL_ENV") or "development"
    return Settings(_env_file=f".env.{environment}")
