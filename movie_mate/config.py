from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    project_root: Path
    openai_api_key: str | None
    openai_base_url: str
    openai_model: str
    openai_embedding_model: str
    openai_timeout_seconds: float
    openai_max_output_tokens: int
    build_embeddings_on_start: bool
    embeddings_cache_path: Path
    memory_store_path: Path
    omdb_api_key: str | None

    @classmethod
    def from_env(cls, project_root: Path) -> "AppConfig":
        return cls(
            project_root=project_root,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"),
            openai_embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_timeout_seconds=float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "45")),
            openai_max_output_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "500")),
            build_embeddings_on_start=os.environ.get("MOVIEMATE_BUILD_EMBEDDINGS_ON_START", "0") == "1",
            embeddings_cache_path=project_root / "data" / "openai_embeddings.json.gz",
            memory_store_path=project_root / "data" / "user_memory.json",
            omdb_api_key=os.environ.get("OMDB_API_KEY"),
        )

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)

    def capabilities(self) -> dict[str, object]:
        return {
            "openai_enabled": self.openai_enabled,
            "openai_model": self.openai_model if self.openai_enabled else None,
            "embedding_model": self.openai_embedding_model if self.openai_enabled else None,
            "memory_enabled": True,
            "api_ingestion_available": True,
        }
