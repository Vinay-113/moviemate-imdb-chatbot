from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from threading import Lock

from .dataset import Movie
from .openai_client import OpenAIClient


class EmbeddingIndex:
    def __init__(
        self,
        movies: list[Movie],
        client: OpenAIClient | None,
        cache_path: Path,
        model_name: str,
    ) -> None:
        self.movies = movies
        self.client = client
        self.cache_path = cache_path
        self.model_name = model_name
        self.dataset_hash = self._dataset_hash(movies)
        self.vectors: dict[int, list[float]] = {}
        self.query_cache: dict[str, list[float]] = {}
        self.lock = Lock()
        self._load_cache()

    @property
    def ready(self) -> bool:
        return len(self.vectors) == len(self.movies)

    def ensure_ready(self, build_if_missing: bool = False) -> bool:
        if self.ready:
            return True
        if build_if_missing and self.client is not None:
            self._build_cache()
        return self.ready

    def search(self, query: str, candidate_ids: list[int], limit: int) -> list[tuple[int, float]] | None:
        if not self.ready and not self.ensure_ready(build_if_missing=False):
            return None
        query_vector = self._query_vector(query)
        if not query_vector:
            return None
        scored = [
            (movie_id, self._dot(query_vector, self.vectors[movie_id]))
            for movie_id in candidate_ids
            if movie_id in self.vectors
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def similar(self, seed_movie_id: int, candidate_ids: list[int], limit: int) -> list[tuple[int, float]] | None:
        if not self.ready and not self.ensure_ready(build_if_missing=False):
            return None
        seed_vector = self.vectors.get(seed_movie_id)
        if seed_vector is None:
            return None
        scored = [
            (movie_id, self._dot(seed_vector, self.vectors[movie_id]))
            for movie_id in candidate_ids
            if movie_id != seed_movie_id and movie_id in self.vectors
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _query_vector(self, text: str) -> list[float] | None:
        if text in self.query_cache:
            return self.query_cache[text]
        if self.client is None:
            return None
        vector = self.client.create_embeddings([text])[0]
        normalized = self._normalize(vector)
        self.query_cache[text] = normalized
        return normalized

    def _build_cache(self) -> None:
        if self.client is None:
            return
        with self.lock:
            if self.ready:
                return
            texts = [movie.search_text for movie in self.movies]
            vectors: list[list[float]] = []
            batch_size = 50
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                vectors.extend(self.client.create_embeddings(batch))
            self.vectors = {
                movie.movie_id: self._normalize(vector)
                for movie, vector in zip(self.movies, vectors, strict=True)
            }
            self._save_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with gzip.open(self.cache_path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return
        if payload.get("dataset_hash") != self.dataset_hash or payload.get("model") != self.model_name:
            return
        stored_vectors = payload.get("vectors", [])
        if len(stored_vectors) != len(self.movies):
            return
        self.vectors = {
            movie.movie_id: vector for movie, vector in zip(self.movies, stored_vectors, strict=True)
        }

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_hash": self.dataset_hash,
            "model": self.model_name,
            "vectors": [self.vectors[movie.movie_id] for movie in self.movies],
        }
        with gzip.open(self.cache_path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def _dataset_hash(self, movies: list[Movie]) -> str:
        digest = hashlib.sha256()
        for movie in movies:
            digest.update(movie.search_text.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _dot(self, left: list[float], right: list[float]) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
