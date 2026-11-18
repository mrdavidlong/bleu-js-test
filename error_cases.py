"""The error report card: deliberately bad requests and how we grade the replies.

A good API answers a bad request with the right 4xx status, a message that
says what went wrong, and a request id you can quote to support. Each case
below sends one bad request; `grade_case` checks those three things.
"""

from dataclasses import dataclass
from typing import Any

from bleu_client import ApiCallResult, extract_error_message

# Stand-in for "use the real key from the app's secrets" in a case definition.
USE_REAL_KEY = "<real key>"

VALID_MESSAGES = [{"role": "user", "content": "Say hi in three words."}]


@dataclass(frozen=True)
class ErrorCase:
    """One deliberately bad request and the status codes a good API would return."""

    name: str  # short label shown in the report table
    path: str  # endpoint under BASE_URL, e.g. "/chat"
    api_key: str | None  # USE_REAL_KEY, a fake key, or None for "no header"
    expected_statuses: tuple[int, ...]  # any of these counts as the right status
    json_payload: Any = None
    raw_body: str | None = None  # when set, sent verbatim instead of JSON


ERROR_CASES: tuple[ErrorCase, ...] = (
    ErrorCase(
        name="No API key",
        path="/chat",
        api_key=None,
        expected_statuses=(401,),
        json_payload={"messages": VALID_MESSAGES},
    ),
    ErrorCase(
        name="Wrong API key",
        path="/chat",
        api_key="bleujs_sk_not_a_real_key",
        expected_statuses=(401, 403),
        json_payload={"messages": VALID_MESSAGES},
    ),
    ErrorCase(
        name="Empty messages list",
        path="/chat",
        api_key=USE_REAL_KEY,
        expected_statuses=(400, 422),
        json_payload={"messages": []},
    ),
    ErrorCase(
        name="Unknown message role",
        path="/chat",
        api_key=USE_REAL_KEY,
        expected_statuses=(400, 422),
        json_payload={"messages": [{"role": "wizard", "content": "hi"}]},
    ),
    ErrorCase(
        name="Negative max_tokens",
        path="/chat",
        api_key=USE_REAL_KEY,
        expected_statuses=(400, 422),
        json_payload={"messages": VALID_MESSAGES, "max_tokens": -5},
    ),
    ErrorCase(
        name="Unknown model name",
        path="/chat",
        api_key=USE_REAL_KEY,
        expected_statuses=(400, 404, 422),
        json_payload={"messages": VALID_MESSAGES, "model": "no-such-model"},
    ),
    ErrorCase(
        name="Body is not JSON",
        path="/chat",
        api_key=USE_REAL_KEY,
        expected_statuses=(400, 422),
        raw_body="this is not json",
    ),
    ErrorCase(
        name="Embed with empty input",
        path="/embed",
        api_key=USE_REAL_KEY,
        expected_statuses=(400, 422),
        json_payload={"input": []},
    ),
)


@dataclass
class CaseGrade:
    """How the API did on one ErrorCase; one row of the report card."""

    case_name: str
    status_code: int  # what the API actually returned (0 = no response)
    right_status: bool  # status was one of the case's expected_statuses
    error_message: str | None  # the human-readable message, if any
    has_request_id: bool  # response carried an x-request-id header

    @property
    def score(self) -> int:
        """Points out of 3: right status, a message, a request id."""
        return sum(
            [self.right_status, self.error_message is not None, self.has_request_id]
        )


def grade_case(case: ErrorCase, result: ApiCallResult) -> CaseGrade:
    """Grade one API reply against what a well-behaved API should have sent."""
    return CaseGrade(
        case_name=case.name,
        status_code=result.status_code,
        right_status=result.status_code in case.expected_statuses,
        error_message=extract_error_message(result.body),
        has_request_id="x-request-id" in result.headers,
    )


def resolve_api_key(case: ErrorCase, real_api_key: str) -> str | None:
    """Swap the USE_REAL_KEY marker for the real key; leave fake/None keys alone."""
    return real_api_key if case.api_key == USE_REAL_KEY else case.api_key
