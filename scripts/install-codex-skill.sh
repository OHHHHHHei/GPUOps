#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
source_path="$repo_root/skills/gpu-infra"
destination_root="${CODEX_HOME:-$HOME/.codex}/skills"
destination="$destination_root/gpu-infra"

mkdir -p "$destination_root"
if [ -e "$destination" ] || [ -L "$destination" ]; then
  current=$(readlink "$destination" 2>/dev/null || true)
  if [ "$current" = "$source_path" ]; then
    printf '%s\n' "gpu-infra skill is already installed"
    exit 0
  fi
  printf '%s\n' "refusing to replace existing path: $destination" >&2
  exit 1
fi

ln -s "$source_path" "$destination"
printf '%s\n' "installed gpu-infra skill -> $source_path"
