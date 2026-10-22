# Contributing to GPUOps

GPUOps is a CLI-first scheduler for trusted Unix accounts on Linux NVIDIA
servers, controlled from a POSIX machine. See [README](README.md) for setup and
[Architecture](docs/ARCHITECTURE.md) for component and ownership boundaries.

## Develop locally

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
node --test tests/test_web_polling.cjs
uv build --wheel
```

Use Python 3.10+ and Node.js 22+ for the browser polling tests. Linux supervisor
tests exercise actual process behavior with synthetic GPU observations. A local
test pass does not establish live SSH, driver, or CUDA readiness.

Keep changes focused, explain the behavior they change, and include the checks
you ran. Add tests for meaningful behavior changes. UI changes should include a
local browser check or a screenshot when useful to review the result.

## Preserve the integration contract

The CLI and `--json` output are the public integration surface. Read
[CLI contract](docs/CLI_CONTRACT.md) before changing result fields, exit codes,
manifest semantics, or default host selection. Maintain compatibility with
persisted queue records, supervisor receipts, submission journals, and retained
releases. Keep examples and the optional Codex skill consistent with the CLI.

Experiment code, environments, data, metrics, and checkpoints belong in the
experiment project. Keep job submission, allocation, logs, recovery, and
cancellation within GPUOps' existing scope.

## Check a built distribution

When changing packaging, installation, or packaged resources, install the built
wheel in a temporary environment and run it outside the checkout. Check
`gpuops version --json` and the relevant command or packaged skill resource.
When changing the skill, run the skill-creator `quick_validate.py` tool if it is
available and review its links and authorization boundaries.

Remote execution or deployment changes also need the
[remote acceptance procedure](docs/ROADMAP.md#remote-acceptance-gate) on an
authorized test server. Documentation and local Web changes do not require a
Worker rollout or GPU use. Record what was actually checked and leave unavailable
checks unverified.

## Share reproducible reports

Include the GPUOps version, operating systems, relevant command, expected and
observed behavior, and a small manifest or JSON error when relevant. Remove
credentials and private experiment data before sharing logs. Do not commit
server registries, SSH keys, remote state, generated logs, or experiment outputs.
Keep host-specific acceptance records private. Share a redacted summary of
results and limitations without usernames, SSH routes, experiment data, or
process command lines from other users.

Contributions are provided under the project's [MIT License](LICENSE).
