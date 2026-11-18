"""Tests for error_cases: grading a reply and choosing which key each case sends."""

import pytest

from bleu_client import ApiCallResult
from error_cases import (
    ERROR_CASES,
    USE_REAL_KEY,
    ErrorCase,
    grade_case,
    resolve_api_key,
)

EMPTY_LIST_CASE = ErrorCase(
    name="Empty messages list",
    path="/chat",
    api_key=USE_REAL_KEY,
    expected_statuses=(400, 422),
    json_payload={"messages": []},
)


@pytest.mark.parametrize(
    ("result", "expected_score", "expected_right_status"),
    [
        pytest.param(
            ApiCallResult(
                422,
                {"error": {"message": "messages is empty"}},
                5.0,
                {"x-request-id": "r1"},
            ),
            3,
            True,
            id="perfect-reply",
        ),
        pytest.param(
            ApiCallResult(422, {}, 5.0, {}),
            1,
            True,
            id="right-status-only",
        ),
        pytest.param(
            ApiCallResult(500, "", 5.0, {}),
            0,
            False,
            id="server-error-no-message-no-id",
        ),
        pytest.param(
            ApiCallResult(200, {"choices": []}, 5.0, {"x-request-id": "r2"}),
            1,
            False,
            id="accepted-bad-input-with-id",
        ),
        pytest.param(
            ApiCallResult(0, None, 5.0, {}, network_error="ConnectError"),
            0,
            False,
            id="no-response",
        ),
    ],
)
def test_grade_case(result, expected_score, expected_right_status):
    """Replies to an empty-messages request score one point each for status, message, request id."""
    grade = grade_case(EMPTY_LIST_CASE, result)

    assert grade.score == expected_score
    assert grade.right_status is expected_right_status
    assert grade.status_code == result.status_code


@pytest.mark.parametrize(
    ("case_key", "expected"),
    [
        pytest.param(USE_REAL_KEY, "real_key", id="marker-becomes-real-key"),
        pytest.param("bleujs_sk_fake", "bleujs_sk_fake", id="fake-key-kept"),
        pytest.param(None, None, id="no-key-kept"),
    ],
)
def test_resolve_api_key(case_key, expected):
    """The USE_REAL_KEY marker is swapped for the real key; fake and missing keys pass through."""
    case = ErrorCase(name="x", path="/chat", api_key=case_key, expected_statuses=(400,))
    assert resolve_api_key(case, "real_key") == expected


def test_error_cases_are_well_formed():
    """Every built-in case has a unique name, a 4xx expectation, and exactly one body kind."""
    names = [case.name for case in ERROR_CASES]
    assert len(names) == len(set(names))
    for case in ERROR_CASES:
        assert all(400 <= status < 500 for status in case.expected_statuses)
        assert (case.json_payload is None) != (case.raw_body is None)
