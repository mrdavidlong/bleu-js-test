# Bleu.js Playground

A small, unofficial test bench for the [Bleu.js](https://bleujs.org) AI API. It's one
Streamlit page with three tabs:

- **Chat**: talk to the model and see latency, token count, rate-limit headers and the
  request id for each reply.
- **Similarity**: embed a few sentences with `/embed` and compare every pair.
- **Error report card**: send 8 deliberately bad requests and score each reply on
  status code, error message and request id.

The API key stays on the server, and a passcode keeps strangers from spending the free
tier's 500 lifetime calls. Each browser session is also capped at 40 calls.

## Run locally

Requires [uv](https://docs.astral.sh/uv/). Everything installs into `./.venv`, nothing
system-wide.

```sh
uv sync                                                  # create .venv and install deps
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit in your key + a passcode
uv run streamlit run app.py
```

## Test

```sh
uv run pytest      # unit + page smoke tests; no API calls
uv run ruff check .
```

## Deploy (Streamlit Community Cloud)

1. At [share.streamlit.io](https://share.streamlit.io), click **Create app**, pick this
   repo, branch `main`, file `app.py`, and Python 3.11 under Advanced settings.
2. In **Advanced settings → Secrets**, paste the two lines from
   `.streamlit/secrets.toml.example` with real values.
3. Deploy, then share the URL and the passcode.

`requirements.txt` is generated from `uv.lock` for the host. After changing
dependencies, regenerate it:

```sh
uv export --no-dev --no-hashes --no-emit-project -o requirements.txt
```

## Why raw HTTP instead of the `bleujs` SDK?

The SDK (`pip install bleu-js`) returns only the parsed body, so the rate-limit and
request-id headers are lost. It also retries failed calls 3 times by default, which
would spend extra credits on the error tests. `bleu_client.py` calls the HTTP API once
per click and keeps the headers.
