# Contributing

1. Use Python 3.12 or newer and install the locked environment with `uv sync`.
2. Keep training and serving changes behind the model protocol in
   `src/ml_model_service/protocol.py`.
3. Add or update deterministic, network-free tests for behavior changes.
4. Run `uv run ruff format .`, `uv run ruff check .`, `uv run pyright`, and
   `uv run pytest` before opening a change.
5. Do not commit `.env` files, credentials, generated artifacts, or large model
   binaries. The checked-in CSV must remain synthetic and small.

Use focused commits and explain any change to the fixture, labels, model version,
or API contract in the pull request.
