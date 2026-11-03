# Subagents

- Delegate independent, bounded subtasks when parallel work would materially improve speed or quality.
- Keep trivial or tightly coupled work in the main agent, and use the project's default subagent configuration when delegating.

## Cursor Cloud specific instructions

- This VM has no NVIDIA GPU and no registered SSH servers. Local tests use synthetic GPU observations. Node.js 22 and OpenSSH are already on the image. `python3-venv` must be installed or `tests/test_deploy.py` fails while creating a remote-install virtualenv (`ensurepip` is missing).
- Install and checks are in `CONTRIBUTING.md`: `uv sync --all-extras --frozen`, then `uv run ruff check .`, `uv run pytest`, and `node --test tests/test_web_polling.cjs`. `uv` is available as `/usr/local/bin/uv`.
- The web console binds to loopback only. With no fleet, run `uv run gpuops web --demo` and open `http://127.0.0.1:8787`. `uv run gpuops version --json` checks the CLI. Do not start a second console on port 8787 while one is already listening.
