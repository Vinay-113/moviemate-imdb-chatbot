from __future__ import annotations

import csv
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


KNOWN_YEAR_FIXES = {
    # The provided CSV has one malformed year value for Apollo 13.
    "Apollo 13": 1995,
}

STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "are",
    "be",
    "best",
    "but",
    "by",
    "can",
    "find",
    "for",
    "from",
    "give",
    "good",
    "high",
    "highly",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "just",
    "like",
    "list",
    "me",
    "movie",
    "movies",
    "of",
    "on",
    "only",
    "or",
    "please",
    "recommend",
    "released",
    "search",
    "show",
    "similar",
    "some",
    "suggest",
    "than",
    "that",
    "the",
    "them",
    "those",
    "to",
    "under",
    "want",
    "what",
    "which",
    "with",
    "without",
    "year",
}


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Movie:
    movie_id: int
    title: str
    title_key: str
    year: int | None
    certificate: str
    runtime_minutes: int | None
    genres: tuple[str, ...]
    rating: float
    overview: str
    metascore: int | None
    director: str
    stars: tuple[str, ...]
    votes: int
    gross: int | None
    poster_url: str
    search_text: str


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_key(text: str) -> str:
    lowered = strip_accents(text.lower())
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_key(text)
    return [
        token
        for token in TOKEN_RE.findall(normalized)
        if token not in STOPWORDS and len(token) > 1
    ]


def _parse_optional_int(raw: str) -> int | None:
    cleaned = raw.strip().replace(",", "")
    return int(cleaned) if cleaned else None


def _parse_runtime(raw: str) -> int | None:
    match = re.search(r"(\d+)", raw)
    return int(match.group(1)) if match else None


def _parse_year(raw: str, title: str) -> int | None:
    cleaned = raw.strip()
    if cleaned.isdigit():
        return int(cleaned)
    return KNOWN_YEAR_FIXES.get(title)


def load_movies(csv_path: str | Path) -> list[Movie]:
    path = Path(csv_path)
    movies: list[Movie] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            title = row["Series_Title"].strip()
            genres = tuple(part.strip() for part in row["Genre"].split(",") if part.strip())
            stars = tuple(
                row[key].strip()
                for key in ("Star1", "Star2", "Star3", "Star4")
                if row[key].strip()
            )
            year = _parse_year(row["Released_Year"], title)
            runtime = _parse_runtime(row["Runtime"])
            rating = float(row["IMDB_Rating"])
            metascore = _parse_optional_int(row["Meta_score"])
            votes = int(row["No_of_Votes"].replace(",", ""))
            gross = _parse_optional_int(row["Gross"])
            poster_url = row["Poster_Link"].strip()
            certificate = row["Certificate"].strip() or "Unknown"
            overview = " ".join(row["Overview"].split())
            search_text = " ".join(
                [
                    title,
                    " ".join(genres),
                    row["Director"].strip(),
                    " ".join(stars),
                    overview,
                    str(year or ""),
                ]
            ).strip()

            movies.append(
                Movie(
                    movie_id=index,
                    title=title,
                    title_key=normalize_key(title),
                    year=year,
                    certificate=certificate,
                    runtime_minutes=runtime,
                    genres=genres,
                    rating=rating,
                    overview=overview,
                    metascore=metascore,
                    director=row["Director"].strip(),
                    stars=stars,
                    votes=votes,
                    gross=gross,
                    poster_url=poster_url,
                    search_text=search_text,
                )
            )

    return movies


def _histogram(values: list[float], start: float, end: float, step: float) -> list[dict[str, int | str]]:
    buckets: list[dict[str, int | str]] = []
    cursor = start
    while cursor < end:
        upper = round(cursor + step, 1)
        count = sum(1 for value in values if cursor <= value < upper or (upper == end and value == upper))
        buckets.append({"label": f"{cursor:.1f}-{upper:.1f}", "value": count})
        cursor = upper
    return buckets


def compute_insights(movies: list[Movie]) -> dict[str, object]:
    ratings = [movie.rating for movie in movies]
    metascores = [movie.metascore for movie in movies if movie.metascore is not None]
    runtimes = [movie.runtime_minutes for movie in movies if movie.runtime_minutes is not None]

    genre_counts: Counter[str] = Counter()
    director_counts: Counter[str] = Counter()
    star_counts: Counter[str] = Counter()
    decade_counts: Counter[str] = Counter()

    missing_counts = {
        "Certificate": sum(1 for movie in movies if movie.certificate == "Unknown"),
        "Meta_score": sum(1 for movie in movies if movie.metascore is None),
        "Gross": sum(1 for movie in movies if movie.gross is None),
    }

    for movie in movies:
        genre_counts.update(movie.genres)
        director_counts[movie.director] += 1
        star_counts.update(movie.stars)
        if movie.year is not None:
            decade = f"{movie.year // 10 * 10}s"
            decade_counts[decade] += 1

    return {
        "summary": {
            "movie_count": len(movies),
            "average_rating": round(statistics.mean(ratings), 2),
            "median_rating": round(statistics.median(ratings), 2),
            "average_runtime": round(statistics.mean(runtimes), 1),
            "average_metascore": round(statistics.mean(metascores), 1),
        },
        "missing_fields": [
            {"label": label, "value": value} for label, value in missing_counts.items()
        ],
        "top_genres": [
            {"label": label, "value": value} for label, value in genre_counts.most_common(8)
        ],
        "top_directors": [
            {"label": label, "value": value} for label, value in director_counts.most_common(6)
        ],
        "top_stars": [
            {"label": label, "value": value} for label, value in star_counts.most_common(6)
        ],
        "decades": [
            {"label": label, "value": value}
            for label, value in sorted(decade_counts.items(), key=lambda item: item[0])
        ],
        "rating_distribution": _histogram(ratings, 7.6, 9.4, 0.2),
        "data_notes": [
            "Cleaned 1 malformed release year by restoring Apollo 13 to 1995.",
            "Missing values remain for Gross, Meta_score, and Certificate, and the app handles them safely.",
            "Search features blend structured filters with TF-IDF retrieval over title, genre, cast, director, and overview text.",
        ],
    }
