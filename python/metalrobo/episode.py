"""Capture-to-world compiler frontend."""

from __future__ import annotations

import os
from pathlib import Path

from .native import MetalRoboError, _load_bindings


def compile_episode_manifest(
    manifest_path: str | os.PathLike[str],
    output_pack_path: str | os.PathLike[str],
    *,
    artifact_store_path: str | os.PathLike[str] | None = None,
    library_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Compile one capture manifest into a reusable ``MRWorldPack``.

    Capture files and deterministic stage products are content-addressed in
    ``artifact_store_path``. Repeating the call resumes compatible stages from
    their receipts while the returned pack remains directly loadable by
    :class:`PackedWorldFamily`.
    """

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Capture manifest does not exist: {manifest}"
        )
    output = Path(output_pack_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_store = (
        Path(artifact_store_path).expanduser().resolve()
        if artifact_store_path is not None
        else None
    )
    bindings = _load_bindings(library_path)
    status = bindings.lib.mr_compile_episode_manifest(
        os.fsencode(manifest),
        os.fsencode(output),
        os.fsencode(artifact_store)
        if artifact_store is not None
        else None,
    )
    if status != 0:
        raise MetalRoboError(
            "Could not compile capture manifest: "
            f"{bindings.last_error()}"
        )
    return output
