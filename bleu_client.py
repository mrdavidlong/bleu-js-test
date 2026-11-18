"""Thin HTTP wrapper around the Bleu.js API, used by the playground app.

We call the HTTP API directly with httpx instead of the official `bleujs`
SDK because the SDK hides response headers (rate limit, request id, token
count) and retries failed calls automatically. For a product test we want to
see every header and spend exactly one credit per click.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

BASE_URL = "https://api.bleujs.org/api/v1"

# Seconds to wait for the API before giving up on a single call.
REQUEST_TIMEOUT_SECONDS = 60.0

# Response headers the Bleu.js docs say every call returns; shown in the UI.
INTERESTING_HEADERS = (
    "x-request-id",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-tokens-total",
)


@dataclass
class ApiCallResult:
    """Everything we learned from one HTTP call to the Bleu.js API."""

    status_code: int  # HTTP status; 0 means the request never got a response
    body: Any  # parsed JSON body, or the raw text when the body isn't JSON
    latency_ms: float  # wall-clock time from send to full response
    # Subset of response headers from INTERESTING_HEADERS; key is the
    # lowercased header name, value is the header's raw string value.
    headers: dict[str, str] = field(default_factory=dict)
    network_error: str | None = None  # set when no HTTP response came back

    @property
    def ok(self) -> bool:
        """True for a 2xx response."""
        return 200 <= self.status_code < 300


def call_endpoint(
    path: str,
    api_key: str | None,
    json_payload: Any = None,
    raw_body: str | None = None,
    http_client: httpx.Client | None = None,
) -> ApiCallResult:
    """Send one POST to a Bleu.js endpoint and capture status, body, headers, latency.

    Args:
        path: endpoint path under BASE_URL, e.g. "/chat", "/embed".
        api_key: the Bleu.js secret key; None sends no Authorization header
            (used by the error report card to test the "missing key" case).
        json_payload: request body to send as JSON.
        raw_body: send this exact string instead of JSON (used to test how
            the API handles a malformed body). Ignored when None.
        http_client: optional client to reuse; tests pass one with a mock
            transport. When None, a short-lived client is created.

    Returns:
        An ApiCallResult. HTTP errors (4xx/5xx) are returned, not raised,
        because the UI wants to show them. A timeout or connection failure
        comes back with status_code 0 and `network_error` set.
    """
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    client = http_client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
    started = time.perf_counter()
    try:
        if raw_body is not None:
            response = client.post(BASE_URL + path, headers=headers, content=raw_body)
        else:
            response = client.post(BASE_URL + path, headers=headers, json=json_payload)
    except httpx.HTTPError as exc:
        return ApiCallResult(
            status_code=0,
            body=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            network_error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if http_client is None:
            client.close()

    latency_ms = (time.perf_counter() - started) * 1000
    try:
        body = response.json()
    except ValueError:
        body = response.text
    kept_headers = {
        name: response.headers[name]
        for name in INTERESTING_HEADERS
        if name in response.headers
    }
    return ApiCallResult(
        status_code=response.status_code,
        body=body,
        latency_ms=latency_ms,
        headers=kept_headers,
    )


def extract_chat_text(body: Any) -> str | None:
    """Pull the assistant's reply out of a /chat response body.

    Accepts the documented OpenAI-style shape (choices[0].message.content)
    and a flat {"content": "..."} shape. Returns None when neither matches,
    so the UI can show the raw body and flag the surprise.
    """
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    if isinstance(body.get("content"), str):
        return body["content"]
    return None


def extract_embeddings(body: Any) -> list[list[float]] | None:
    """Pull one vector per input string out of an /embed response body.

    Expects the OpenAI-style shape {"data": [{"embedding": [...]}, ...]}.
    Returns None when the body doesn't match, so the UI can show it raw.
    """
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        return None
    vectors: list[list[float]] = []
    for item in body["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
            return None
        vectors.append([float(value) for value in item["embedding"]])
    return vectors


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Cosine similarity of two vectors: 1.0 same direction, 0.0 unrelated.

    Raises:
        ValueError: when the vectors have different lengths. A zero-length or
            all-zero vector returns 0.0 rather than dividing by zero, since
            "no signal" is the honest answer for an empty embedding.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"vectors differ in length: {len(vector_a)} vs {len(vector_b)}"
        )
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    return dot_product / (norm_a * norm_b)


def similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """Pairwise cosine similarity for every pair of vectors (row i, column j)."""
    return [[cosine_similarity(row, column) for column in vectors] for row in vectors]


def extract_error_message(body: Any) -> str | None:
    """Find a human-readable error message in an error response body.

    Handles {"error": {"message": ...}}, {"error": "..."}, {"message": ...},
    and FastAPI-style {"detail": ...}. Returns None when there is none,
    which the error report card counts against the API.
    """
    if isinstance(body, str):
        return body.strip() or None
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if isinstance(error, str) and error:
        return error
    for key in ("message", "detail"):
        if body.get(key):
            return str(body[key])
    return None


def preserve_line_breaks(text: str) -> str:
    """Make every newline in a chat reply show up as a line break in Markdown.

    Streamlit renders chat text as Markdown, where a single newline is folded
    into the surrounding paragraph. Bleu.js replies put each "•" point on its
    own line, so without this they run together into one paragraph. Ending a
    line with two spaces is Markdown's "hard line break".
    """
    return text.replace("\n", "  \n")


def conversation_length(messages: list[dict[str, str]]) -> int:
    """Total characters across all message contents in a chat conversation.

    The free (trial) plan rejects a /chat call whose messages add up to more
    than 1,000 characters (error code `plan_prompt_length_exceeded`), and the
    whole conversation is resent on every turn, so the UI shows this count.
    Each message is a {"role": ..., "content": ...} dict; a missing content
    counts as 0.
    """
    return sum(len(message.get("content", "")) for message in messages)
