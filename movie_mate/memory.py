from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


@dataclass(slots=True)
class UserProfile:
    profile_id: str
    query_count: int = 0
    genre_counts: Counter[str] = field(default_factory=Counter)
    person_counts: Counter[str] = field(default_factory=Counter)
    director_counts: Counter[str] = field(default_factory=Counter)
    recent_queries: list[str] = field(default_factory=list)
    recent_titles: list[str] = field(default_factory=list)
    last_updated: str | None = None

    @classmethod
    def from_payload(cls, profile_id: str, payload: dict[str, object]) -> "UserProfile":
        return cls(
            profile_id=profile_id,
            query_count=int(payload.get("query_count", 0)),
            genre_counts=Counter(payload.get("genre_counts", {})),
            person_counts=Counter(payload.get("person_counts", {})),
            director_counts=Counter(payload.get("director_counts", {})),
            recent_queries=list(payload.get("recent_queries", [])),
            recent_titles=list(payload.get("recent_titles", [])),
            last_updated=payload.get("last_updated"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "query_count": self.query_count,
            "genre_counts": dict(self.genre_counts),
            "person_counts": dict(self.person_counts),
            "director_counts": dict(self.director_counts),
            "recent_queries": self.recent_queries[-12:],
            "recent_titles": self.recent_titles[-12:],
            "last_updated": self.last_updated,
        }


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = Lock()
        self.profiles = self._load_profiles()

    def get_profile(self, profile_id: str | None) -> UserProfile | None:
        if not profile_id:
            return None
        with self.lock:
            return self.profiles.setdefault(profile_id, UserProfile(profile_id=profile_id))

    def summarize(self, profile: UserProfile | None) -> str:
        if profile is None or profile.query_count == 0:
            return "No saved user history yet."

        parts: list[str] = []
        if profile.genre_counts:
            top_genres = ", ".join(
                f"{label} ({count})" for label, count in profile.genre_counts.most_common(3)
            )
            parts.append(f"Preferred genres: {top_genres}.")
        if profile.director_counts:
            top_directors = ", ".join(
                f"{label} ({count})" for label, count in profile.director_counts.most_common(2)
            )
            parts.append(f"Frequent directors: {top_directors}.")
        if profile.person_counts:
            top_people = ", ".join(
                f"{label} ({count})" for label, count in profile.person_counts.most_common(3)
            )
            parts.append(f"Frequent actors/people: {top_people}.")
        if profile.recent_titles:
            parts.append(f"Recently explored: {', '.join(profile.recent_titles[-5:])}.")
        return " ".join(parts)

    def update_profile(
        self,
        profile_id: str | None,
        message: str,
        genres: list[str],
        people: list[str],
        directors: list[str],
        titles: list[str],
    ) -> None:
        if not profile_id:
            return

        with self.lock:
            profile = self.profiles.setdefault(profile_id, UserProfile(profile_id=profile_id))
            profile.query_count += 1
            profile.genre_counts.update(genres)
            profile.person_counts.update(people)
            profile.director_counts.update(directors)
            profile.recent_queries.append(message)
            profile.recent_queries = profile.recent_queries[-12:]
            for title in titles:
                if title in profile.recent_titles:
                    profile.recent_titles.remove(title)
                profile.recent_titles.append(title)
            profile.recent_titles = profile.recent_titles[-12:]
            profile.last_updated = datetime.now(timezone.utc).isoformat()
            self._save_profiles()

    def personalization_bonus(
        self,
        profile: UserProfile | None,
        movie_genres: tuple[str, ...],
        director: str,
        stars: tuple[str, ...],
    ) -> float:
        if profile is None or profile.query_count == 0:
            return 0.0

        bonus = 0.0
        bonus += sum(profile.genre_counts.get(genre, 0) for genre in movie_genres) * 0.01
        bonus += profile.director_counts.get(director, 0) * 0.03
        bonus += sum(profile.person_counts.get(star, 0) for star in stars) * 0.015
        return min(bonus, 0.18)

    def _load_profiles(self) -> dict[str, UserProfile]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw_profiles = payload.get("profiles", {})
        return {
            profile_id: UserProfile.from_payload(profile_id, profile_payload)
            for profile_id, profile_payload in raw_profiles.items()
        }

    def _save_profiles(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "profiles": {
                profile_id: profile.to_payload() for profile_id, profile in self.profiles.items()
            }
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
