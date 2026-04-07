from __future__ import annotations

from pathlib import Path

from movie_mate import AppConfig, EmbeddingIndex, OpenAIClient, OpenAISettings, load_movies


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config = AppConfig.from_env(project_root)
    if not config.openai_enabled:
        raise SystemExit("OPENAI_API_KEY is required to build the embedding cache.")

    client = OpenAIClient(
        OpenAISettings(
            api_key=config.openai_api_key or "",
            base_url=config.openai_base_url,
            model=config.openai_model,
            embedding_model=config.openai_embedding_model,
            timeout_seconds=config.openai_timeout_seconds,
            max_output_tokens=config.openai_max_output_tokens,
        )
    )
    movies = load_movies(project_root / "data" / "imdb_top_1000.csv")
    index = EmbeddingIndex(
        movies=movies,
        client=client,
        cache_path=config.embeddings_cache_path,
        model_name=config.openai_embedding_model,
    )
    index.ensure_ready(build_if_missing=True)
    print(f"Embedding cache ready at {config.embeddings_cache_path}")


if __name__ == "__main__":
    main()
