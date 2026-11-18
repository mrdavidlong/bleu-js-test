"""Bleu.js Playground: a small Streamlit app for trying out the Bleu.js API.

Three tabs:
  * Chat: send a message, see the reply plus latency, tokens and rate-limit headers.
  * Similarity: embed a few sentences and compare them pairwise.
  * Error report card: send deliberately bad requests and grade the replies.

The API key lives in server-side secrets (never in the browser). A passcode
gate keeps strangers from spending the key's limited free credits.

Run locally:  uv run streamlit run app.py
"""

import hmac
import json
import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from bleu_client import (
    call_endpoint,
    extract_chat_text,
    extract_embeddings,
    preserve_line_breaks,
    similarity_matrix,
)
from error_cases import ERROR_CASES, grade_case, resolve_api_key

# The free tier is 500 lifetime calls, so cap how many one browser session
# can spend. A reload resets this; the passcode is the real protection.
MAX_CALLS_PER_SESSION = 40

DEFAULT_SENTENCES = (
    "The cat sat on the mat.\n"
    "A kitten is resting on the rug.\n"
    "The car needs new tires.\n"
    "The automobile needs new tyres."
)


def read_setting(name: str) -> str | None:
    """Read a setting from an environment variable, then from Streamlit secrets.

    Returns None when it is set in neither place. A missing secrets.toml file
    is normal for env-var-only setups, so that case is not an error.
    """
    if os.environ.get(name):
        return os.environ[name]
    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None
    return str(value) if value else None


def require_passcode(expected_passcode: str) -> None:
    """Show a passcode box and stop rendering the page until it is entered correctly."""
    if st.session_state.get("unlocked"):
        return
    entered = st.text_input("Passcode", type="password")
    if entered and hmac.compare_digest(entered, expected_passcode):
        st.session_state.unlocked = True
        st.rerun()
    if entered:
        st.error("Wrong passcode.")
    st.stop()


def spend_calls(count: int) -> bool:
    """Reserve `count` API calls from this session's budget.

    Returns False (and shows a warning) when the budget would be exceeded,
    in which case nothing is spent and the caller must not make the calls.
    """
    used = st.session_state.get("calls_used", 0)
    if used + count > MAX_CALLS_PER_SESSION:
        st.warning(
            f"This session has used {used} of {MAX_CALLS_PER_SESSION} calls. "
            "Reload the page to reset."
        )
        return False
    st.session_state.calls_used = used + count
    return True


def show_call_stats(result) -> None:
    """Show latency, status and the Bleu.js headers for one API call."""
    columns = st.columns(4)
    columns[0].metric("Status", result.status_code or "no response")
    columns[1].metric("Latency", f"{result.latency_ms:,.0f} ms")
    columns[2].metric("Tokens", result.headers.get("x-tokens-total", "n/a"))
    rate_limit = result.headers.get("x-ratelimit-limit")
    remaining = result.headers.get("x-ratelimit-remaining")
    columns[3].metric(
        "Rate limit left", f"{remaining}/{rate_limit}" if rate_limit else "n/a"
    )
    if result.headers.get("x-request-id"):
        st.caption(f"Request id: `{result.headers['x-request-id']}`")


def chat_tab(api_key: str) -> None:
    """Chat playground: a conversation with the model, with per-reply stats."""
    st.subheader("Chat")
    max_tokens = st.slider("max_tokens", min_value=16, max_value=1024, value=200)
    model = st.text_input(
        "Model (leave blank for the API default)",
        help="The homepage example says `bleujs-chat`; the Python SDK defaults to `bleu-chat-v1`.",
    )

    history = st.session_state.setdefault("chat_history", [])
    for message in history:
        st.chat_message(message["role"]).markdown(
            preserve_line_breaks(message["content"])
        )

    prompt = st.chat_input("Say something to Bleu.js")
    if not prompt or not spend_calls(1):
        return

    history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    payload = {"messages": history, "max_tokens": max_tokens}
    if model.strip():
        payload["model"] = model.strip()

    with st.spinner("Waiting for Bleu.js..."):
        result = call_endpoint("/chat", api_key, json_payload=payload)

    reply = extract_chat_text(result.body) if result.ok else None
    if reply is not None:
        history.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").markdown(preserve_line_breaks(reply))
    else:
        # Drop the unanswered user turn so the next try doesn't resend it twice.
        history.pop()
        st.error(result.network_error or "The API did not return a chat reply.")
        st.json(
            result.body
            if isinstance(result.body, (dict, list))
            else {"body": result.body}
        )
    show_call_stats(result)


def similarity_tab(api_key: str) -> None:
    """Embed a few sentences and show how similar each pair is."""
    st.subheader("Similarity explorer")
    st.write(
        "One sentence per line. Bleu.js describes its embeddings as *lexical* "
        "(word-based), so expect synonyms like car/automobile to score low."
    )
    text = st.text_area("Sentences", value=DEFAULT_SENTENCES, height=150)
    sentences = [line.strip() for line in text.splitlines() if line.strip()]
    if not st.button("Compare", disabled=len(sentences) < 2):
        return
    if not spend_calls(1):
        return

    with st.spinner("Embedding..."):
        result = call_endpoint("/embed", api_key, json_payload={"input": sentences})
    show_call_stats(result)

    vectors = extract_embeddings(result.body) if result.ok else None
    if vectors is None or len(vectors) != len(sentences):
        st.error(result.network_error or "Unexpected /embed response.")
        st.json(
            result.body
            if isinstance(result.body, (dict, list))
            else {"body": result.body}
        )
        return

    st.caption(f"{len(vectors)} vectors of length {len(vectors[0])}")
    short_labels = [
        f"{index + 1}. {sentence[:30]}" for index, sentence in enumerate(sentences)
    ]
    matrix = similarity_matrix(vectors)
    st.dataframe(
        {
            label: [round(row[column], 3) for row in matrix]
            for column, label in enumerate(short_labels)
        },
        hide_index=False,
    )
    st.caption(
        "Rows and columns are in the same order as the sentences. 1.0 = identical."
    )


def error_report_tab(api_key: str) -> None:
    """Send each deliberately bad request and grade how well the API explains it."""
    st.subheader("Error report card")
    st.write(
        f"Sends {len(ERROR_CASES)} deliberately bad requests, one call each. "
        "Each reply scores a point for the right status code, a readable message, "
        "and a request id."
    )
    if not st.button(f"Run all {len(ERROR_CASES)} checks"):
        return
    if not spend_calls(len(ERROR_CASES)):
        return

    rows = []
    total_score = 0
    with st.spinner("Probing..."):
        for case in ERROR_CASES:
            result = call_endpoint(
                case.path,
                resolve_api_key(case, api_key),
                json_payload=case.json_payload,
                raw_body=case.raw_body,
            )
            grade = grade_case(case, result)
            total_score += grade.score
            rows.append(
                {
                    "Check": grade.case_name,
                    "Expected": " or ".join(
                        str(code) for code in case.expected_statuses
                    ),
                    "Got": grade.status_code,
                    "Right status": "yes" if grade.right_status else "no",
                    "Message": grade.error_message or "(none)",
                    "Request id": "yes" if grade.has_request_id else "no",
                    "Score": f"{grade.score}/3",
                }
            )
    st.metric("Total", f"{total_score}/{3 * len(rows)}")
    st.dataframe(rows, hide_index=True)
    st.download_button(
        "Download as JSON (to send as feedback)",
        data=json.dumps(rows, indent=2),
        file_name="bleujs-error-report.json",
        mime="application/json",
    )


def main() -> None:
    """Page entry point: check config, gate on the passcode, render the tabs."""
    st.set_page_config(page_title="Bleu.js Playground", page_icon="🔵")
    st.title("Bleu.js Playground")
    st.caption("An unofficial test bench for the [Bleu.js](https://bleujs.org) API.")

    api_key = read_setting("BLEUJS_API_KEY")
    passcode = read_setting("APP_PASSCODE")
    if not api_key or not passcode:
        st.error(
            "Missing configuration: set BLEUJS_API_KEY and APP_PASSCODE in "
            ".streamlit/secrets.toml (locally) or in the app's Secrets settings "
            "(Streamlit Cloud)."
        )
        st.stop()
    require_passcode(passcode)

    st.caption(
        f"Calls used this session: {st.session_state.get('calls_used', 0)} / "
        f"{MAX_CALLS_PER_SESSION}"
    )
    chat, similarity, errors = st.tabs(["Chat", "Similarity", "Error report card"])
    with chat:
        chat_tab(api_key)
    with similarity:
        similarity_tab(api_key)
    with errors:
        error_report_tab(api_key)


main()
