"""The release and stable interface versions exposed to Agent clients."""

from gpuops import __version__

CLI_API_VERSION = "gpuops.cli/v1"


def version_info() -> dict[str, str]:
    from gpuops.worker.rpc import RPC_API_VERSION

    return {
        "version": __version__,
        "cli_api_version": CLI_API_VERSION,
        "job_api_version": "gpuops/v1",
        "worker_api_version": RPC_API_VERSION,
    }
