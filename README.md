# MovieMate

`MovieMate` is a conversational movie exploration project built from the provided IMDb Top 1000 CSV. It covers the assignment requirements with:

- dataset loading and cleaning
- summary insights for EDA
- lexical retrieval fallback using TF-IDF
- optional OpenAI Responses API integration for natural-language replies
- optional OpenAI embedding-based semantic retrieval with a local vector cache
- persistent per-browser memory based on stored user history
- conversational follow-up filtering
- title-detail lookups and "similar to" search
- optional OMDb API dataset acquisition/enrichment pipeline
- a local web interface built with the Python standard library
- a notebook-style deliverable for submission

## Live Demo

[https://moviemate-imdb-chatbot.onrender.com](https://moviemate-imdb-chatbot.onrender.com)

## Run the app

```bash
cd /Users/vinaypatil/Documents/Playground/imdb_movie_chatbot
python3 app.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Optional OpenAI setup

Set these environment variables to enable the OpenAI-backed path:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-5.4-mini"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
```

If you want semantic embedding retrieval enabled in the app, build the cache once:

```bash
cd /Users/vinaypatil/Documents/Playground/imdb_movie_chatbot
python3 build_embeddings.py
```

Then start the app normally. The UI shows whether it is running in fallback mode, OpenAI mode, or OpenAI RAG mode.

If you prefer automatic embedding generation at startup:

```bash
export MOVIEMATE_BUILD_EMBEDDINGS_ON_START=1
```

## Run tests

```bash
cd /Users/vinaypatil/Documents/Playground/imdb_movie_chatbot
python3 -m unittest discover -s tests -v
```

## Public deployment on Render

This project is prepared for Render web service deployment.

Files:

- `render.yaml`
- `requirements.txt`

Render settings are already encoded to:

- use the Python runtime
- run `pip install -r requirements.txt`
- start the app with `python3 app.py --host 0.0.0.0 --port $PORT`
- use `/api/insights` as the health check
- optionally accept `OPENAI_API_KEY` and `OMDB_API_KEY` as Render secrets

If you want the public deployment to use OpenAI replies:

1. Open the Render service settings.
2. Add `OPENAI_API_KEY` as an environment variable.
3. Optionally set `MOVIEMATE_BUILD_EMBEDDINGS_ON_START=1` for automatic semantic cache generation.
4. Redeploy the service.

Note: Render's free filesystem is ephemeral, so rebuilding embeddings on startup is fine for a prototype but not ideal for production.

To publish it:

1. Push this folder to a GitHub repository.
2. In Render, create a new `Web Service` from that repo, or use the blueprint from `render.yaml`.
3. After the first deploy finishes, Render will assign an `onrender.com` URL you can share.

## Dataset acquisition pipeline

An API-based dataset builder is included for OMDb:

```bash
cd /Users/vinaypatil/Documents/Playground/imdb_movie_chatbot
export OMDB_API_KEY="your-omdb-key"
python3 scripts/acquire_dataset_from_omdb.py --titles-file /path/to/titles.txt --output data/omdb_movies.csv
```

You can also point it at an existing CSV with a `Series_Title` column:

```bash
python3 scripts/acquire_dataset_from_omdb.py --source-csv data/imdb_top_1000.csv --output data/omdb_enriched.csv
```

## Project layout

- `app.py`: app entry point
- `build_embeddings.py`: builds the OpenAI embedding cache
- `movie_mate/dataset.py`: dataset loading, cleaning, and insights
- `movie_mate/chatbot.py`: query parsing, retrieval, personalization, and response orchestration
- `movie_mate/openai_client.py`: OpenAI Responses and Embeddings API client
- `movie_mate/rag.py`: embedding cache and semantic retrieval
- `movie_mate/memory.py`: persistent user preference memory
- `movie_mate/server.py`: local HTTP server and API endpoints
- `scripts/acquire_dataset_from_omdb.py`: OMDb-powered dataset acquisition/enrichment
- `static/`: frontend files
- `notebooks/imdb_chatbot_analysis.ipynb`: notebook deliverable
- `tests/`: regression tests

## Notes

- The provided dataset had one malformed release year for `Apollo 13`; preprocessing restores it to `1995`.
- Missing values in `Gross`, `Meta_score`, and `Certificate` are preserved and handled safely.
- The app always works without API keys by falling back to deterministic retrieval and template responses.
- OpenAI-powered responses and embeddings become active only when the relevant environment variables are configured.
