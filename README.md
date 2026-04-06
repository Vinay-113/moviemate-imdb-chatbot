# MovieMate

`MovieMate` is a conversational movie exploration project built from the provided IMDb Top 1000 CSV. It covers the assignment requirements with:

- dataset loading and cleaning
- summary insights for EDA
- retrieval-based movie search using TF-IDF
- conversational follow-up filtering
- title-detail lookups and "similar to" search
- a local web interface built with the Python standard library
- a notebook-style deliverable for submission

## Run the app

```bash
cd /Users/vinaypatil/Documents/Playground/imdb_movie_chatbot
python3 app.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

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

To publish it:

1. Push this folder to a GitHub repository.
2. In Render, create a new `Web Service` from that repo, or use the blueprint from `render.yaml`.
3. After the first deploy finishes, Render will assign an `onrender.com` URL you can share.

## Project layout

- `app.py`: app entry point
- `movie_mate/dataset.py`: dataset loading, cleaning, and insights
- `movie_mate/chatbot.py`: query parsing, retrieval, and conversation state
- `movie_mate/server.py`: local HTTP server and API endpoints
- `static/`: frontend files
- `notebooks/imdb_chatbot_analysis.ipynb`: notebook deliverable
- `tests/`: regression tests

## Notes

- The provided dataset had one malformed release year for `Apollo 13`; preprocessing restores it to `1995`.
- Missing values in `Gross`, `Meta_score`, and `Certificate` are preserved and handled safely.
- The response layer is deterministic and retrieval-first, which keeps the app fully runnable in this environment without external dependencies or API keys.
