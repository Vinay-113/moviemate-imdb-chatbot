from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


class OpenAIAPIError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenAISettings:
    api_key: str
    base_url: str
    model: str
    embedding_model: str
    timeout_seconds: float
    max_output_tokens: int


class OpenAIClient:
    def __init__(self, settings: OpenAISettings) -> None:
        self.settings = settings

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
        }
        response = self._post_json("/embeddings", payload)
        items = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding", []) for item in items]
        if len(embeddings) != len(texts):
            raise OpenAIAPIError("Embedding response did not include vectors for every input.")
        return embeddings

    def generate_text(self, instructions: str, prompt: str) -> str:
        payload = {
            "model": self.settings.model,
            "instructions": instructions,
            "input": prompt,
            "max_output_tokens": self.settings.max_output_tokens,
        }
        response = self._post_json("/responses", payload)
        text = response.get("output_text") or self._extract_output_text(response)
        if not text:
            raise OpenAIAPIError("Responses API returned no text output.")
        return text.strip()

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.settings.base_url}{path}",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - exercised only with real API traffic
            detail = exc.read().decode("utf-8", "ignore")
            raise OpenAIAPIError(f"OpenAI API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - exercised only with real API traffic
            raise OpenAIAPIError(f"OpenAI API request failed: {exc.reason}") from exc

    def _extract_output_text(self, response: dict[str, object]) -> str:
        chunks: list[str] = []
        for item in response.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text" and content.get("text"):
                    chunks.append(str(content["text"]))
        return "\n".join(chunks).strip()
