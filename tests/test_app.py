"""Smoke tests for the Streamlit page: config check and passcode gate.

Uses Streamlit's AppTest runner; none of these paths call the Bleu.js API.
"""

import pytest
from streamlit.testing.v1 import AppTest

APP_FILE = "../app.py"  # relative to this test file


@pytest.fixture
def configured_env(monkeypatch):
    """Provide a fake key and a known passcode through environment variables."""
    monkeypatch.setenv("BLEUJS_API_KEY", "bleujs_sk_test")
    monkeypatch.setenv("APP_PASSCODE", "letmein")


def test_missing_config_shows_setup_error(monkeypatch):
    """No key or passcode configured: the page shows a setup error and no tabs."""
    monkeypatch.delenv("BLEUJS_API_KEY", raising=False)
    monkeypatch.delenv("APP_PASSCODE", raising=False)
    app = AppTest.from_file(APP_FILE)
    app.secrets["unused"] = "forces an empty secrets store"
    app.run()

    assert "Missing configuration" in app.error[0].value
    assert len(app.tabs) == 0


def test_wrong_passcode_keeps_page_locked(configured_env):
    """Passcode 'nope' when 'letmein' is expected: error shown, tabs stay hidden."""
    app = AppTest.from_file(APP_FILE).run()
    app.text_input[0].input("nope").run()

    assert app.error[0].value == "Wrong passcode."
    assert len(app.tabs) == 0


def test_right_passcode_unlocks_three_tabs(configured_env):
    """Correct passcode: the page shows the Chat, Similarity and Error report card tabs."""
    app = AppTest.from_file(APP_FILE).run()
    app.text_input[0].input("letmein").run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Chat",
        "Similarity",
        "Error report card",
    ]
