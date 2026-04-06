from __future__ import annotations

import argparse
import os
from pathlib import Path

from movie_mate import MovieChatbot, compute_insights, load_movies
from movie_mate.server import build_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MovieMate IMDb chatbot.")
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Host to bind the local server to.",
    )
    parser.add_argument(
        "--port",
        default=int(os.environ.get("PORT", "8000")),
        type=int,
        help="Port for the local server.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "data" / "imdb_top_1000.csv"
    static_dir = project_root / "static"

    movies = load_movies(csv_path)
    chatbot = MovieChatbot(movies)
    insights = compute_insights(movies)

    server = build_server(args.host, args.port, chatbot, insights, static_dir)
    print(f"MovieMate is running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MovieMate...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
