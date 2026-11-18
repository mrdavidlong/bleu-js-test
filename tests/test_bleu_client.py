"""Tests for bleu_client: the HTTP wrapper, response parsers, and similarity math.

HTTP calls go through httpx.MockTransport, so no real API credits are spent.
"""

import json

import httpx
import pytest

from bleu_client import (
    BASE_URL,
    call_endpoint,
    conversation_length,
    cosine_similarity,
    extract_chat_text,
    extract_embeddings,
    extract_error_message,
    preserve_line_breaks,
    similarity_matrix,
    similarity_table_rows,
)


def client_returning(handler) -> httpx.Client:
    """An httpx client whose every request is answered by `handler`."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_successful_call_captures_status_body_headers_and_latency():
    """A 200 with Bleu.js headers: result is ok, body parsed, only known headers kept."""
    seen_requests = []

    def handler(request):
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={"hello": "world"},
            headers={
                "x-request-id": "req_1",
                "x-ratelimit-remaining": "59",
                "server": "x",
            },
        )

    result = call_endpoint(
        "/chat", "key_123", {"a": 1}, http_client=client_returning(handler)
    )

    assert result.ok
    assert result.body == {"hello": "world"}
    assert result.headers == {"x-request-id": "req_1", "x-ratelimit-remaining": "59"}
    assert result.latency_ms >= 0
    assert str(seen_requests[0].url) == BASE_URL + "/chat"
    assert seen_requests[0].headers["Authorization"] == "Bearer key_123"
    assert json.loads(seen_requests[0].content) == {"a": 1}


def test_missing_key_sends_no_authorization_header():
    """api_key=None: the request goes out with no Authorization header at all."""
    seen_requests = []

    def handler(request):
        seen_requests.append(request)
        return httpx.Response(401, json={"error": {"message": "no key"}})

    result = call_endpoint("/chat", None, {}, http_client=client_returning(handler))

    assert "Authorization" not in seen_requests[0].headers
    assert result.status_code == 401
    assert not result.ok


def test_raw_body_is_sent_verbatim_and_text_reply_is_kept_as_text():
    """raw_body="not json": sent unchanged; a plain-text reply comes back as a string."""
    seen_requests = []

    def handler(request):
        seen_requests.append(request)
        return httpx.Response(400, text="bad body")

    result = call_endpoint(
        "/chat", "k", raw_body="not json", http_client=client_returning(handler)
    )

    assert seen_requests[0].content == b"not json"
    assert result.body == "bad body"


def test_network_failure_returns_status_zero_with_error():
    """Connection refused: no exception escapes; status 0 and network_error is set."""

    def handler(request):
        raise httpx.ConnectError("refused")

    result = call_endpoint("/chat", "k", {}, http_client=client_returning(handler))

    assert result.status_code == 0
    assert result.body is None
    assert "ConnectError" in result.network_error


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            {"choices": [{"message": {"content": "hi"}}]}, "hi", id="openai-shape"
        ),
        pytest.param({"content": "flat hi"}, "flat hi", id="flat-shape"),
        pytest.param({"choices": []}, None, id="empty-choices"),
        pytest.param(
            {"choices": [{"message": {}}]}, None, id="message-without-content"
        ),
        pytest.param("just text", None, id="not-a-dict"),
        pytest.param(None, None, id="none"),
    ],
)
def test_extract_chat_text(body, expected):
    """Chat bodies of each known shape give the reply text; anything else gives None."""
    assert extract_chat_text(body) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            {"data": [{"embedding": [1, 2]}, {"embedding": [3.5, 4]}]},
            [[1.0, 2.0], [3.5, 4.0]],
            id="two-vectors",
        ),
        pytest.param({"data": []}, [], id="empty-data"),
        pytest.param({"data": [{"vector": [1]}]}, None, id="item-missing-embedding"),
        pytest.param({"result": []}, None, id="no-data-key"),
        pytest.param("oops", None, id="not-a-dict"),
    ],
)
def test_extract_embeddings(body, expected):
    """Embed bodies in the {data: [{embedding}]} shape give vectors; others give None."""
    assert extract_embeddings(body) == expected


@pytest.mark.parametrize(
    ("vector_a", "vector_b", "expected"),
    [
        pytest.param([1, 2, 3], [1, 2, 3], 1.0, id="identical"),
        pytest.param([1, 0], [0, 1], 0.0, id="orthogonal"),
        pytest.param([1, 1], [-1, -1], -1.0, id="opposite"),
        pytest.param([0, 0], [1, 1], 0.0, id="zero-vector"),
        pytest.param([], [], 0.0, id="empty-vectors"),
    ],
)
def test_cosine_similarity(vector_a, vector_b, expected):
    """Known vector pairs give the textbook cosine; zero/empty vectors give 0.0."""
    assert cosine_similarity(vector_a, vector_b) == pytest.approx(expected)


def test_cosine_similarity_rejects_different_lengths():
    """Vectors of length 2 and 3: raises ValueError naming both lengths."""
    with pytest.raises(ValueError, match="2 vs 3"):
        cosine_similarity([1, 2], [1, 2, 3])


def test_similarity_matrix_is_symmetric_with_unit_diagonal():
    """Three vectors: 3x3 matrix, 1.0 on the diagonal, [i][j] == [j][i]."""
    matrix = similarity_matrix([[1, 0], [0, 1], [1, 1]])

    assert len(matrix) == 3
    for index in range(3):
        assert matrix[index][index] == pytest.approx(1.0)
    assert matrix[0][2] == pytest.approx(matrix[2][0])
    assert matrix[0][1] == pytest.approx(0.0)


def test_similarity_matrix_single_vector():
    """One vector: a 1x1 matrix containing 1.0."""
    assert similarity_matrix([[3, 4]]) == [[pytest.approx(1.0)]]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param({"error": {"message": "bad key"}}, "bad key", id="nested-error"),
        pytest.param({"error": "bad key"}, "bad key", id="error-string"),
        pytest.param({"message": "bad key"}, "bad key", id="top-level-message"),
        pytest.param(
            {"detail": [{"msg": "field required"}]},
            "[{'msg': 'field required'}]",
            id="fastapi-detail",
        ),
        pytest.param("  plain text  ", "plain text", id="text-body"),
        pytest.param("", None, id="empty-text"),
        pytest.param({"error": {}}, None, id="empty-error-object"),
        pytest.param(None, None, id="none"),
    ],
)
def test_extract_error_message(body, expected):
    """Each common error-body shape yields its message; empty or unknown shapes yield None."""
    assert extract_error_message(body) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "Intro.\n\n• one\n• two",
            "Intro.  \n  \n• one  \n• two",
            id="bleu-bullet-reply",
        ),
        pytest.param("no newlines", "no newlines", id="single-line"),
        pytest.param("", "", id="empty"),
    ],
)
def test_preserve_line_breaks(text, expected):
    """Each newline becomes a Markdown hard break so "•" lines don't merge into one paragraph."""
    assert preserve_line_breaks(text) == expected


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param([], 0, id="empty-conversation"),
        pytest.param([{"role": "user", "content": "Hello!"}], 6, id="one-message"),
        pytest.param(
            [
                {"role": "user", "content": "abc"},
                {"role": "assistant", "content": "x" * 997},
            ],
            1000,
            id="exactly-at-free-limit",
        ),
        pytest.param([{"role": "user"}], 0, id="missing-content"),
    ],
)
def test_conversation_length(messages, expected):
    """Sums content characters across all turns; roles and missing content add nothing."""
    assert conversation_length(messages) == expected


def test_similarity_table_rows_label_both_axes():
    """Two sentences: each row starts with its 1-based label, scores rounded to 3 places."""
    rows = similarity_table_rows(["1. cat", "2. car"], [[1.0, 0.98765], [0.98765, 1.0]])

    assert rows == [
        {"Sentence": "1. cat", "1. cat": 1.0, "2. car": 0.988},
        {"Sentence": "2. car", "1. cat": 0.988, "2. car": 1.0},
    ]


def test_similarity_table_rows_rejects_mismatched_labels():
    """Three labels for a 2x2 matrix: raises ValueError instead of a misaligned table."""
    with pytest.raises(ValueError, match="3 labels for a 2-row matrix"):
        similarity_table_rows(["a", "b", "c"], [[1.0, 0.0], [0.0, 1.0]])
