from __future__ import annotations

import math
import re
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field

from .dataset import Movie, normalize_key, tokenize


FOLLOW_UP_MARKERS = (
    "only",
    "those",
    "them",
    "ones",
    "what about",
    "how about",
    "also",
    "instead",
)

RESET_MARKERS = (
    "show me",
    "recommend",
    "suggest",
    "find",
    "list",
    "search",
    "i want",
    "give me",
)

DETAIL_PATTERNS = {
    "director": ("who directed", "director of"),
    "cast": ("who stars in", "cast of", "who is in"),
    "rating": ("rating of", "imdb rating", "how highly rated"),
    "runtime": ("runtime of", "how long is", "duration of"),
    "year": ("release year", "when was", "released in what year"),
    "overview": ("what is", "tell me about", "overview of", "plot of"),
}


@dataclass(slots=True)
class SearchPlan:
    raw_query: str
    semantic_query: str
    genres: set[str] = field(default_factory=set)
    people: set[str] = field(default_factory=set)
    directors: set[str] = field(default_factory=set)
    year_min: int | None = None
    year_max: int | None = None
    runtime_min: int | None = None
    runtime_max: int | None = None
    rating_min: float | None = None
    rating_max: float | None = None
    count: int = 5
    sort_by: str = "relevance"
    similar_to: str | None = None
    title_reference: str | None = None
    wants_details: bool = False
    wants_count: bool = False
    specific_detail: str | None = None
    filter_only: bool = False

    def clone(self) -> "SearchPlan":
        return SearchPlan(
            raw_query=self.raw_query,
            semantic_query=self.semantic_query,
            genres=set(self.genres),
            people=set(self.people),
            directors=set(self.directors),
            year_min=self.year_min,
            year_max=self.year_max,
            runtime_min=self.runtime_min,
            runtime_max=self.runtime_max,
            rating_min=self.rating_min,
            rating_max=self.rating_max,
            count=self.count,
            sort_by=self.sort_by,
            similar_to=self.similar_to,
            title_reference=self.title_reference,
            wants_details=self.wants_details,
            wants_count=self.wants_count,
            specific_detail=self.specific_detail,
            filter_only=self.filter_only,
        )


@dataclass(slots=True)
class SessionState:
    last_plan: SearchPlan | None = None
    last_results: list[int] = field(default_factory=list)


class MovieChatbot:
    def __init__(self, movies: list[Movie]) -> None:
        self.movies = movies
        self.movie_by_id = {movie.movie_id: movie for movie in movies}
        self.title_index = self._build_title_index(movies)
        self.sorted_title_keys = sorted(self.title_index, key=len, reverse=True)
        self.genre_aliases = self._build_genre_aliases(movies)
        self.name_index = self._build_name_index(movies)
        self.sorted_name_keys = sorted(self.name_index, key=len, reverse=True)
        self.document_vectors, self.document_norms, self.idf = self._build_index(movies)
        self.sessions: dict[str, SessionState] = {}
        self.lock = threading.Lock()

    def new_session_id(self) -> str:
        return uuid.uuid4().hex

    def respond(self, message: str, session_id: str | None = None) -> dict[str, object]:
        clean_message = " ".join(message.split())
        if not clean_message:
            clean_message = "Show me highly rated movies."

        with self.lock:
            active_session_id = session_id or self.new_session_id()
            session = self.sessions.setdefault(active_session_id, SessionState())

            parsed_plan = self._parse_query(clean_message)
            used_context = False
            if session.last_plan and self._should_use_context(clean_message, parsed_plan):
                parsed_plan = self._merge_with_context(session.last_plan, parsed_plan)
                used_context = True

            if parsed_plan.wants_details and parsed_plan.title_reference:
                movie = self._find_title(parsed_plan.title_reference)
                results = [movie] if movie else []
                reply = self._build_detail_reply(movie, parsed_plan)
            else:
                results = self._search(parsed_plan)
                reply = self._build_search_reply(results, parsed_plan, used_context)

            session.last_plan = parsed_plan
            session.last_results = [movie.movie_id for movie in results]

        return {
            "session_id": active_session_id,
            "reply": reply,
            "cards": [self._movie_card(movie) for movie in results[: parsed_plan.count]],
            "used_context": used_context,
            "filters": self._summarize_filters(parsed_plan),
        }

    def _build_title_index(self, movies: list[Movie]) -> dict[str, list[Movie]]:
        index: dict[str, list[Movie]] = {}
        for movie in movies:
            index.setdefault(movie.title_key, []).append(movie)
        return index

    def _build_genre_aliases(self, movies: list[Movie]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for movie in movies:
            for genre in movie.genres:
                aliases[normalize_key(genre)] = genre
        aliases.update(
            {
                "sci fi": "Sci-Fi",
                "science fiction": "Sci-Fi",
                "scifi": "Sci-Fi",
                "film noir": "Film-Noir",
                "rom com": "Comedy",
                "biopic": "Biography",
            }
        )
        return aliases

    def _build_name_index(self, movies: list[Movie]) -> dict[str, tuple[str, str]]:
        index: dict[str, tuple[str, str]] = {}
        for movie in movies:
            key = normalize_key(movie.director)
            index[key] = ("director", movie.director)
            for star in movie.stars:
                star_key = normalize_key(star)
                index[star_key] = ("person", star)
        return index

    def _build_index(
        self, movies: list[Movie]
    ) -> tuple[list[dict[str, float]], list[float], dict[str, float]]:
        document_term_counts: list[Counter[str]] = []
        document_frequencies: Counter[str] = Counter()

        for movie in movies:
            counts: Counter[str] = Counter()
            self._add_weighted_tokens(counts, movie.title, weight=4)
            self._add_weighted_tokens(counts, " ".join(movie.genres), weight=3)
            self._add_weighted_tokens(counts, movie.director, weight=2)
            self._add_weighted_tokens(counts, " ".join(movie.stars), weight=2)
            self._add_weighted_tokens(counts, movie.overview, weight=1)
            if movie.year is not None:
                counts[str(movie.year)] += 1
            document_term_counts.append(counts)
            document_frequencies.update(counts.keys())

        total_documents = len(movies)
        idf = {
            token: 1.0 + math.log((1 + total_documents) / (1 + frequency))
            for token, frequency in document_frequencies.items()
        }

        vectors: list[dict[str, float]] = []
        norms: list[float] = []
        for counts in document_term_counts:
            vector = {token: count * idf[token] for token, count in counts.items()}
            norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
            vectors.append(vector)
            norms.append(norm)
        return vectors, norms, idf

    def _add_weighted_tokens(self, counter: Counter[str], text: str, weight: int) -> None:
        for token in tokenize(text):
            counter[token] += weight

    def _parse_query(self, query: str) -> SearchPlan:
        query_key = normalize_key(query)
        plan = SearchPlan(raw_query=query, semantic_query=query)
        semantic_tokens = tokenize(query)
        plan.count = self._extract_count(query_key)
        plan.wants_count = "how many" in query_key or "count " in query_key

        for detail_key, markers in DETAIL_PATTERNS.items():
            if any(marker in query_key for marker in markers):
                plan.wants_details = True
                plan.specific_detail = detail_key
                break

        if any(token in query_key for token in ("tell me about", "details on", "details of", "information on")):
            plan.wants_details = True

        if "similar to" in query_key or query_key.startswith("movies like ") or " like " in query_key:
            title_match = self._extract_title_from_query(query_key)
            if title_match:
                plan.similar_to = title_match.title
                plan.title_reference = title_match.title

        title_reference = self._extract_title_from_query(query_key, require_trigger=plan.wants_details)
        if title_reference:
            plan.title_reference = title_reference.title

        plan.genres = self._extract_genres(query_key)
        people, directors = self._extract_names(query_key)
        plan.people = people
        plan.directors = directors
        self._extract_year_constraints(query_key, plan)
        self._extract_runtime_constraints(query_key, plan)
        self._extract_rating_constraints(query_key, plan)
        self._extract_sort_preferences(query_key, plan)

        recognized_terms = set(plan.genres) | plan.people | plan.directors
        plan.filter_only = bool(recognized_terms or plan.year_min or plan.year_max or plan.runtime_min or plan.runtime_max)
        if plan.filter_only and plan.sort_by == "relevance" and len(semantic_tokens) <= 4:
            plan.sort_by = "rating"
        return plan

    def _extract_count(self, query_key: str) -> int:
        match = re.search(r"\b(?:top|show|give|list|recommend|suggest)\s+(\d{1,2})\b", query_key)
        if not match:
            match = re.search(r"\b(\d{1,2})\s+(?:movies|films|results|recommendations)\b", query_key)
        if match:
            return max(1, min(12, int(match.group(1))))
        return 5

    def _extract_genres(self, query_key: str) -> set[str]:
        found: set[str] = set()
        padded = f" {query_key} "
        for alias, canonical in self.genre_aliases.items():
            if f" {alias} " in padded:
                found.add(canonical)
        return found

    def _extract_names(self, query_key: str) -> tuple[set[str], set[str]]:
        people: set[str] = set()
        directors: set[str] = set()
        padded = f" {query_key} "
        for name_key in self.sorted_name_keys:
            if f" {name_key} " not in padded:
                continue
            kind, display_name = self.name_index[name_key]
            if kind == "director" and ("directed by" in query_key or "director" in query_key or " by " in padded):
                directors.add(display_name)
            elif kind == "director" and "starring" not in query_key:
                people.add(display_name)
            else:
                people.add(display_name)
        return people, directors

    def _extract_year_constraints(self, query_key: str, plan: SearchPlan) -> None:
        between = re.search(r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b", query_key)
        if between:
            plan.year_min = int(between.group(1))
            plan.year_max = int(between.group(2))
            return

        after = re.search(r"\b(?:after|since|newer than)\s+(\d{4})\b", query_key)
        if after:
            plan.year_min = int(after.group(1)) + 1

        before = re.search(r"\b(?:before|older than)\s+(\d{4})\b", query_key)
        if before:
            plan.year_max = int(before.group(1)) - 1

        exact = re.search(r"\b(?:in|from|released in)\s+(\d{4})\b", query_key)
        if exact and plan.year_min is None and plan.year_max is None:
            value = int(exact.group(1))
            plan.year_min = value
            plan.year_max = value

    def _extract_runtime_constraints(self, query_key: str, plan: SearchPlan) -> None:
        under = re.search(r"\b(?:under|below|less than)\s+(\d{2,3})\s*(?:minutes|mins|min)?\b", query_key)
        if under:
            plan.runtime_max = int(under.group(1)) - 1

        over = re.search(r"\b(?:over|above|more than)\s+(\d{2,3})\s*(?:minutes|mins|min)?\b", query_key)
        if over:
            plan.runtime_min = int(over.group(1)) + 1

    def _extract_rating_constraints(self, query_key: str, plan: SearchPlan) -> None:
        minimum = re.search(r"\b(?:rating|ratings|rated)\s+(?:above|over|at least|min(?:imum)?)\s+(\d(?:\.\d)?)\b", query_key)
        if minimum:
            plan.rating_min = float(minimum.group(1))

        maximum = re.search(r"\b(?:rating|ratings|rated)\s+(?:below|under|at most|max(?:imum)?)\s+(\d(?:\.\d)?)\b", query_key)
        if maximum:
            plan.rating_max = float(maximum.group(1))

        if any(phrase in query_key for phrase in ("best", "top rated", "highly rated", "must watch", "good ")):
            plan.rating_min = max(plan.rating_min or 0.0, 8.0)
            if plan.sort_by == "relevance":
                plan.sort_by = "rating"

    def _extract_sort_preferences(self, query_key: str, plan: SearchPlan) -> None:
        if "newest" in query_key or "latest" in query_key:
            plan.sort_by = "year_desc"
        elif "oldest" in query_key or "classic" in query_key:
            plan.sort_by = "year_asc"
        elif "most popular" in query_key or "most voted" in query_key:
            plan.sort_by = "votes"
        elif "highest rated" in query_key or "top rated" in query_key:
            plan.sort_by = "rating"
        elif "shortest" in query_key:
            plan.sort_by = "runtime_asc"
        elif "longest" in query_key:
            plan.sort_by = "runtime_desc"

    def _extract_title_from_query(self, query_key: str, require_trigger: bool = False) -> Movie | None:
        if require_trigger and not any(
            trigger in query_key
            for trigger in (
                "about",
                "details",
                "similar to",
                "like ",
                "who directed",
                "cast of",
                "runtime of",
                "rating of",
                "overview of",
                "plot of",
                "tell me about",
                "details on",
                "details of",
                "information on",
            )
        ):
            return None

        padded = f" {query_key} "
        for title_key in self.sorted_title_keys:
            if f" {title_key} " not in padded:
                continue
            movies = self.title_index[title_key]
            return sorted(movies, key=lambda movie: movie.votes, reverse=True)[0]

        if query_key in self.title_index:
            movies = self.title_index[query_key]
            return sorted(movies, key=lambda movie: movie.votes, reverse=True)[0]
        return None

    def _should_use_context(self, query: str, plan: SearchPlan) -> bool:
        query_key = normalize_key(query)
        if plan.similar_to or plan.title_reference and plan.wants_details:
            return False
        if any(marker in query_key for marker in RESET_MARKERS):
            return False
        has_follow_up_marker = any(marker in query_key for marker in FOLLOW_UP_MARKERS)
        if has_follow_up_marker:
            return True
        if plan.genres or plan.people or plan.directors:
            return False
        return len(query_key.split()) <= 6 and plan.filter_only

    def _merge_with_context(self, previous: SearchPlan, current: SearchPlan) -> SearchPlan:
        merged = previous.clone()
        merged.raw_query = current.raw_query
        merged.count = current.count or previous.count
        merged.wants_count = current.wants_count
        merged.wants_details = current.wants_details
        merged.specific_detail = current.specific_detail

        if current.genres:
            merged.genres = set(current.genres)
        if current.people:
            merged.people = set(current.people)
        if current.directors:
            merged.directors = set(current.directors)
        if current.year_min is not None:
            merged.year_min = current.year_min
        if current.year_max is not None:
            merged.year_max = current.year_max
        if current.runtime_min is not None:
            merged.runtime_min = current.runtime_min
        if current.runtime_max is not None:
            merged.runtime_max = current.runtime_max
        if current.rating_min is not None:
            merged.rating_min = current.rating_min
        if current.rating_max is not None:
            merged.rating_max = current.rating_max
        if current.sort_by != "relevance":
            merged.sort_by = current.sort_by
        if current.similar_to:
            merged.similar_to = current.similar_to
        if current.title_reference:
            merged.title_reference = current.title_reference
        if current.semantic_query and not current.filter_only:
            merged.semantic_query = current.semantic_query
        return merged

    def _search(self, plan: SearchPlan) -> list[Movie]:
        candidates = [movie for movie in self.movies if self._matches_filters(movie, plan)]
        if not candidates:
            return []

        if plan.wants_count:
            return self._sort_movies(candidates, plan.sort_by)[: plan.count]

        if plan.similar_to:
            seed = self._find_title(plan.similar_to)
            if seed is None:
                return []
            scored = []
            seed_vector = self.document_vectors[seed.movie_id]
            seed_norm = self.document_norms[seed.movie_id]
            for movie in candidates:
                if movie.movie_id == seed.movie_id:
                    continue
                similarity = self._cosine(seed_vector, seed_norm, self.document_vectors[movie.movie_id], self.document_norms[movie.movie_id])
                if plan.genres and any(genre in movie.genres for genre in plan.genres):
                    similarity += 0.08
                if seed.director == movie.director:
                    similarity += 0.07
                shared_stars = len(set(seed.stars) & set(movie.stars))
                similarity += shared_stars * 0.03
                similarity += movie.rating / 100
                scored.append((similarity, movie))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [movie for _, movie in scored[: plan.count]]

        query_vector, query_norm = self._build_query_vector(self._query_text_for_plan(plan))
        scored = []
        for movie in candidates:
            relevance = self._cosine(query_vector, query_norm, self.document_vectors[movie.movie_id], self.document_norms[movie.movie_id])
            if plan.genres:
                relevance += 0.06 * sum(1 for genre in plan.genres if genre in movie.genres)
            if plan.people:
                relevance += 0.07 * sum(1 for person in plan.people if person in movie.stars or person == movie.director)
            if plan.directors and movie.director in plan.directors:
                relevance += 0.08
            relevance += movie.rating / 200
            relevance += min(movie.votes / 2_000_000, 0.08)
            scored.append((relevance, movie))

        if plan.sort_by == "relevance":
            scored.sort(key=lambda item: item[0], reverse=True)
            return [movie for _, movie in scored[: plan.count]]

        sorted_movies = self._sort_movies([movie for _, movie in scored], plan.sort_by)
        return sorted_movies[: plan.count]

    def _query_text_for_plan(self, plan: SearchPlan) -> str:
        pieces = [plan.semantic_query]
        if plan.genres:
            pieces.append(" ".join(plan.genres))
        if plan.people:
            pieces.append(" ".join(plan.people))
        if plan.directors:
            pieces.append(" ".join(plan.directors))
        if plan.similar_to:
            pieces.append(plan.similar_to)
        return " ".join(piece for piece in pieces if piece)

    def _build_query_vector(self, text: str) -> tuple[dict[str, float], float]:
        counts = Counter(tokenize(text))
        if not counts:
            return {}, 1.0
        vector = {}
        for token, count in counts.items():
            idf = self.idf.get(token, 1.0 + math.log(1 + len(self.movies)))
            vector[token] = count * idf
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return vector, norm

    def _cosine(
        self,
        left_vector: dict[str, float],
        left_norm: float,
        right_vector: dict[str, float],
        right_norm: float,
    ) -> float:
        if not left_vector or not right_vector:
            return 0.0
        if len(left_vector) > len(right_vector):
            left_vector, right_vector = right_vector, left_vector
            left_norm, right_norm = right_norm, left_norm
        dot_product = sum(value * right_vector.get(token, 0.0) for token, value in left_vector.items())
        return dot_product / (left_norm * right_norm)

    def _sort_movies(self, movies: list[Movie], sort_by: str) -> list[Movie]:
        if sort_by == "rating":
            return sorted(movies, key=lambda movie: (movie.rating, movie.votes), reverse=True)
        if sort_by == "year_desc":
            return sorted(movies, key=lambda movie: (movie.year or 0, movie.rating), reverse=True)
        if sort_by == "year_asc":
            return sorted(movies, key=lambda movie: (movie.year or 9999, -movie.rating))
        if sort_by == "votes":
            return sorted(movies, key=lambda movie: movie.votes, reverse=True)
        if sort_by == "runtime_asc":
            return sorted(movies, key=lambda movie: movie.runtime_minutes or 9999)
        if sort_by == "runtime_desc":
            return sorted(movies, key=lambda movie: movie.runtime_minutes or 0, reverse=True)
        return sorted(movies, key=lambda movie: (movie.rating, movie.votes), reverse=True)

    def _matches_filters(self, movie: Movie, plan: SearchPlan) -> bool:
        if plan.genres and not plan.genres.issubset(set(movie.genres)):
            return False
        if plan.people and not any(person in movie.stars or person == movie.director for person in plan.people):
            return False
        if plan.directors and movie.director not in plan.directors:
            return False
        if plan.year_min is not None and (movie.year is None or movie.year < plan.year_min):
            return False
        if plan.year_max is not None and (movie.year is None or movie.year > plan.year_max):
            return False
        if plan.runtime_min is not None and (movie.runtime_minutes is None or movie.runtime_minutes < plan.runtime_min):
            return False
        if plan.runtime_max is not None and (movie.runtime_minutes is None or movie.runtime_minutes > plan.runtime_max):
            return False
        if plan.rating_min is not None and movie.rating < plan.rating_min:
            return False
        if plan.rating_max is not None and movie.rating > plan.rating_max:
            return False
        return True

    def _find_title(self, title: str) -> Movie | None:
        key = normalize_key(title)
        if key in self.title_index:
            return sorted(self.title_index[key], key=lambda movie: movie.votes, reverse=True)[0]
        return None

    def _build_detail_reply(self, movie: Movie | None, plan: SearchPlan) -> str:
        if movie is None:
            return "I could not find that title in the IMDb Top 1000 dataset. Try using the exact movie name from the dataset."

        if plan.specific_detail == "director":
            return f"{movie.title} ({movie.year or 'Unknown'}) was directed by {movie.director}."
        if plan.specific_detail == "cast":
            return f"{movie.title} stars {', '.join(movie.stars)}."
        if plan.specific_detail == "rating":
            return f"{movie.title} has an IMDb rating of {movie.rating}/10 from {movie.votes:,} votes."
        if plan.specific_detail == "runtime":
            runtime = f"{movie.runtime_minutes} minutes" if movie.runtime_minutes is not None else "an unknown runtime"
            return f"{movie.title} runs for {runtime}."
        if plan.specific_detail == "year":
            return f"{movie.title} was released in {movie.year or 'an unknown year'}."

        runtime = f"{movie.runtime_minutes} min" if movie.runtime_minutes is not None else "runtime unavailable"
        metascore = movie.metascore if movie.metascore is not None else "N/A"
        return (
            f"{movie.title} ({movie.year or 'Unknown'}) is a {', '.join(movie.genres)} film directed by "
            f"{movie.director}. It is rated {movie.rating}/10 on IMDb, has a metascore of {metascore}, "
            f"runs {runtime}, and stars {', '.join(movie.stars)}. Overview: {movie.overview}"
        )

    def _build_search_reply(self, results: list[Movie], plan: SearchPlan, used_context: bool) -> str:
        if plan.wants_count:
            description = self._describe_filters(plan)
            count = len([movie for movie in self.movies if self._matches_filters(movie, plan)])
            return f"I found {count} movies{description} in the dataset."

        if not results:
            return (
                "I could not find a clean match for that combination of filters. Try broadening the year, "
                "runtime, or rating constraints, or ask for a specific genre, actor, or director."
            )

        lead = "I refined the previous search." if used_context else "Here are some strong matches."
        if plan.similar_to:
            lead = f"If you liked {plan.similar_to}, these are the closest matches from the dataset."

        titles = ", ".join(f"{movie.title} ({movie.year or 'Unknown'})" for movie in results[:3])
        description = self._describe_filters(plan)
        return f"{lead}{description} Top picks include {titles}."

    def _describe_filters(self, plan: SearchPlan) -> str:
        parts: list[str] = []
        if plan.genres:
            parts.append(" in " + ", ".join(sorted(plan.genres)))
        if plan.people:
            parts.append(" starring " + ", ".join(sorted(plan.people)))
        if plan.directors:
            parts.append(" directed by " + ", ".join(sorted(plan.directors)))
        if plan.year_min is not None and plan.year_max is not None and plan.year_min == plan.year_max:
            parts.append(f" from {plan.year_min}")
        else:
            if plan.year_min is not None:
                parts.append(f" after {plan.year_min - 1}")
            if plan.year_max is not None:
                parts.append(f" before {plan.year_max + 1}")
        if plan.rating_min is not None:
            parts.append(f" with IMDb rating >= {plan.rating_min:g}")
        if plan.runtime_max is not None:
            parts.append(f" under {plan.runtime_max + 1} minutes")
        if plan.runtime_min is not None:
            parts.append(f" over {plan.runtime_min - 1} minutes")
        return "".join(parts)

    def _summarize_filters(self, plan: SearchPlan) -> dict[str, object]:
        return {
            "genres": sorted(plan.genres),
            "people": sorted(plan.people),
            "directors": sorted(plan.directors),
            "year_min": plan.year_min,
            "year_max": plan.year_max,
            "runtime_min": plan.runtime_min,
            "runtime_max": plan.runtime_max,
            "rating_min": plan.rating_min,
            "rating_max": plan.rating_max,
            "sort_by": plan.sort_by,
        }

    def _movie_card(self, movie: Movie) -> dict[str, object]:
        return {
            "title": movie.title,
            "year": movie.year,
            "rating": movie.rating,
            "genres": list(movie.genres),
            "runtime_minutes": movie.runtime_minutes,
            "director": movie.director,
            "stars": list(movie.stars),
            "votes": movie.votes,
            "metascore": movie.metascore,
            "overview": movie.overview,
            "certificate": movie.certificate,
            "gross": movie.gross,
            "poster_url": movie.poster_url,
        }
