from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


HEADERS = [
    "Poster_Link",
    "Series_Title",
    "Released_Year",
    "Certificate",
    "Runtime",
    "Genre",
    "IMDB_Rating",
    "Overview",
    "Meta_score",
    "Director",
    "Star1",
    "Star2",
    "Star3",
    "Star4",
    "No_of_Votes",
    "Gross",
]


def fetch_omdb(api_key: str, title: str, year: str | None = None) -> dict[str, object] | None:
    params = {"apikey": api_key, "t": title}
    if year:
        params["y"] = year
    url = "https://www.omdbapi.com/?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("Response") != "True":
        return None
    return payload


def split_stars(raw: str) -> list[str]:
    stars = [part.strip() for part in raw.split(",") if part.strip()]
    return (stars + ["", "", "", ""])[:4]


def normalize_row(payload: dict[str, object]) -> dict[str, str]:
    star1, star2, star3, star4 = split_stars(str(payload.get("Actors", "")))
    return {
        "Poster_Link": str(payload.get("Poster", "")),
        "Series_Title": str(payload.get("Title", "")),
        "Released_Year": str(payload.get("Year", "")).split("–", maxsplit=1)[0],
        "Certificate": str(payload.get("Rated", "")),
        "Runtime": str(payload.get("Runtime", "")),
        "Genre": str(payload.get("Genre", "")),
        "IMDB_Rating": str(payload.get("imdbRating", "")),
        "Overview": str(payload.get("Plot", "")),
        "Meta_score": str(payload.get("Metascore", "")).replace("N/A", ""),
        "Director": str(payload.get("Director", "")),
        "Star1": star1,
        "Star2": star2,
        "Star3": star3,
        "Star4": star4,
        "No_of_Votes": str(payload.get("imdbVotes", "")),
        "Gross": str(payload.get("BoxOffice", "")).replace("$", ""),
    }


def read_title_rows(titles_file: Path | None, source_csv: Path | None) -> list[tuple[str, str | None]]:
    if source_csv:
        with source_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [
                (row.get("Series_Title", "").strip(), row.get("Released_Year", "").strip() or None)
                for row in reader
                if row.get("Series_Title", "").strip()
            ]
    if titles_file:
        lines = [line.strip() for line in titles_file.read_text(encoding="utf-8").splitlines()]
        return [(line, None) for line in lines if line]
    raise SystemExit("Provide either --titles-file or --source-csv.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or enrich a movie dataset using the OMDb API.")
    parser.add_argument("--titles-file", type=Path, help="Plain text file with one movie title per line.")
    parser.add_argument("--source-csv", type=Path, help="Existing CSV with a Series_Title column.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--api-key", default=os.environ.get("OMDB_API_KEY"), help="OMDb API key.")
    parser.add_argument("--delay-seconds", type=float, default=0.2, help="Delay between API calls.")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Provide OMDB_API_KEY or pass --api-key.")

    title_rows = read_title_rows(args.titles_file, args.source_csv)
    output_rows: list[dict[str, str]] = []

    for index, (title, year) in enumerate(title_rows, start=1):
        payload = fetch_omdb(args.api_key, title, year=year)
        if payload is None:
            print(f"[skip] {index}: {title}")
            continue
        output_rows.append(normalize_row(payload))
        print(f"[ok] {index}: {title}")
        time.sleep(args.delay_seconds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} movies to {args.output}")


if __name__ == "__main__":
    main()
