"""Use generic test servers in an isolated registry."""

import json
from dataclasses import asdict

import pytest

from gpuops.config import ServerTarget

TEST_TARGETS = (
    ServerTarget(id="gpu01", label="GPU 01", ssh_host="lab-gpu01"),
    ServerTarget(id="gpu02", label="GPU 02", ssh_host="lab-gpu02"),
)


@pytest.fixture(autouse=True)
def server_registry(tmp_path_factory, monkeypatch):
    path = tmp_path_factory.mktemp("gpuops-config") / "servers.json"
    path.write_text(json.dumps({"servers": [asdict(target) for target in TEST_TARGETS]}))
    monkeypatch.setattr("gpuops.config.registry_path", lambda: path)
    return path
