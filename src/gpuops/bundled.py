"""Read the Agent guides and build metadata carried by every GPUOps distribution."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

# Source paths are also the layout restored when deploying an installed client.
_PROJECT_FILES = ("pyproject.toml", "README.md", "LICENSE")
_AGENT_FILES = {
    "SKILL.md": "skills/gpu-infra/SKILL.md",
    "docs/AGENT_GUIDE.md": "docs/AGENT_GUIDE.md",
    "docs/CLI_CONTRACT.md": "docs/CLI_CONTRACT.md",
    "docs/PROGRESS.md": "docs/PROGRESS.md",
    "examples/job.example.json": "examples/job.example.json",
    "examples/progress.example.py": "examples/progress.example.py",
}


def _read(group: str, name: str, source_path: str) -> str:
    resource = files("gpuops").joinpath("_bundled", group, *name.split("/"))
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    # Editable installs read the same source files that hatch includes in wheels.
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "src/gpuops/bundled.py").resolve() == Path(__file__).resolve():
        source = source_root / source_path
        if source.is_file():
            return source.read_text(encoding="utf-8")
    raise FileNotFoundError(f"GPUOps distribution is missing bundled resource: {source_path}")


def agent_files() -> dict[str, str]:
    """Return the portable skill and its guides, keyed by skill-relative paths."""
    return {
        name: _read("agent", name, source_path)
        for name, source_path in _AGENT_FILES.items()
    }


def project_files() -> dict[str, str]:
    """Return the source-tree files needed to build this installed version."""
    return {
        **{name: _read("project", name, name) for name in _PROJECT_FILES},
        **{
            source_path: _read("agent", name, source_path)
            for name, source_path in _AGENT_FILES.items()
        },
    }
