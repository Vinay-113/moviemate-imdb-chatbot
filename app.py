from __future__ import annotations

import argparse
import os
from pathlib import Path

from movie_mate import (
    AppConfig,
    EmbeddingIndex,
    MemoryStore,
    MovieChatbot,
    OpenAIClient,
    OpenAISettings,
    compute_insights,
    load_movies,
)
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
    config = AppConfig.from_env(project_root)

    movies = load_movies(csv_path)
    openai_client = None
    if config.openai_enabled:
        openai_client = OpenAIClient(
            OpenAISettings(
                api_key=config.openai_api_key or "",
                base_url=config.openai_base_url,
                model=config.openai_model,
                embedding_model=config.openai_embedding_model,
                timeout_seconds=config.openai_timeout_seconds,
                max_output_tokens=config.openai_max_output_tokens,
            )
        )
    embedding_index = EmbeddingIndex(
        movies=movies,
        client=openai_client,
        cache_path=config.embeddings_cache_path,
        model_name=config.openai_embedding_model,
    )
    if config.build_embeddings_on_start:
        embedding_index.ensure_ready(build_if_missing=True)
    memory_store = MemoryStore(config.memory_store_path)
    chatbot = MovieChatbot(
        movies,
        openai_client=openai_client,
        embedding_index=embedding_index,
        memory_store=memory_store,
        auto_build_embeddings=config.build_embeddings_on_start,
    )
    insights = compute_insights(movies)
    insights["capabilities"] = config.capabilities()
    insights["capabilities"]["embedding_cache_ready"] = embedding_index.ready
    insights["capabilities"]["memory_store_path"] = str(config.memory_store_path.name)

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
