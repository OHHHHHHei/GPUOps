"""The release and stable interface versions exposed to Agent clients."""

from gpuops import __version__

CLI_API_VERSION = "gpuops.cli/v1"


def release_mismatch_message(
    client_release: str | None, worker_release: str | None, server_id: str
) -> str:
    client = client_release or "unknown"
    worker = worker_release or "unknown"
    return (
        f"GPUOps versions must match (client: {client}, worker: {worker}). "
        f"Run `gpuops deploy install --server {server_id}`."
    )


def version_info() -> dict[str, str]:
    from gpuops.worker.rpc import RPC_API_VERSION

    return {
        "version": __version__,
        "cli_api_version": CLI_API_VERSION,
        "job_api_version": "gpuops/v1",
        "worker_api_version": RPC_API_VERSION,
    }
