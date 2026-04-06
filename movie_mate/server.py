from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .chatbot import MovieChatbot


class MovieMateRequestHandler(BaseHTTPRequestHandler):
    chatbot: MovieChatbot
    insights: dict[str, object]
    static_dir: Path

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route in {"", "/"}:
            self._serve_file(self.static_dir / "index.html", "text/html; charset=utf-8")
            return

        if route == "/api/insights":
            self._send_json(self.insights)
            return

        if route.startswith("/static/"):
            file_path = self.static_dir / route.removeprefix("/static/")
            self._serve_file(file_path)
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw_body or "{}")
            message = str(payload.get("message", "")).strip()
            session_id = payload.get("session_id")
            response = self.chatbot.respond(message, session_id=session_id)
            self._send_json(response)
        except json.JSONDecodeError:
            self._send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON payload")
        except Exception as exc:  # pragma: no cover - defensive server guard
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def log_message(self, format: str, *args: object) -> None:
        return

    def _serve_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        guessed_type, _ = mimetypes.guess_type(path.name)
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or guessed_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_server(
    host: str,
    port: int,
    chatbot: MovieChatbot,
    insights: dict[str, object],
    static_dir: Path,
) -> ThreadingHTTPServer:
    handler = type(
        "ConfiguredMovieMateRequestHandler",
        (MovieMateRequestHandler,),
        {"chatbot": chatbot, "insights": insights, "static_dir": static_dir},
    )
    return ThreadingHTTPServer((host, port), handler)
