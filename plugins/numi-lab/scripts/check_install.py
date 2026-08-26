#!/usr/bin/env python3
"""Verify that Codex has loaded the exact local Numi Lab plugin source."""

from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path
import sys


PLUGIN_ID = "numi-lab@numi-lab"


def files(root: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != ".DS_Store"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    arguments = parser.parse_args()

    manifest_path = arguments.source / ".codex-plugin" / "plugin.json"
    try:
        expected_version = json.loads(manifest_path.read_text())["version"]
        listing = json.load(sys.stdin)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"numi codex: could not inspect plugin metadata: {error}", file=sys.stderr)
        return 1

    plugin = next(
        (
            item
            for item in listing.get("installed", [])
            if item.get("pluginId") == PLUGIN_ID and item.get("installed") is True
        ),
        None,
    )
    if plugin is None:
        print("Numi Lab is not installed. Run `numi codex install`.", file=sys.stderr)
        return 2
    if plugin.get("enabled") is not True:
        print("Numi Lab is installed but disabled; reinstall or enable it.", file=sys.stderr)
        return 3

    installed_version = plugin.get("version")
    if installed_version != expected_version:
        print(
            "Numi Lab is stale: Codex loaded "
            f"{installed_version!s}, source requires {expected_version}. "
            "Run `numi codex install`.",
            file=sys.stderr,
        )
        return 4

    loaded_source = Path(plugin.get("source", {}).get("path", "")).resolve()
    if loaded_source != arguments.source.resolve():
        print(
            f"Numi Lab source mismatch: Codex reports {loaded_source}; "
            f"this runtime owns {arguments.source.resolve()}.",
            file=sys.stderr,
        )
        return 5

    cache = arguments.cache_root / "numi-lab" / "numi-lab" / expected_version
    if not cache.is_dir():
        print(f"Numi Lab cache is missing: {cache}", file=sys.stderr)
        return 6

    source_files = files(arguments.source)
    cache_files = files(cache)
    if source_files.keys() != cache_files.keys() or any(
        not filecmp.cmp(source_files[name], cache_files[name], shallow=False)
        for name in source_files.keys() & cache_files.keys()
    ):
        print(
            "Numi Lab cache content differs from the plugin source. "
            "Run `numi codex install`.",
            file=sys.stderr,
        )
        return 7

    print(f"Numi Lab is installed, enabled, and current ({expected_version}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
