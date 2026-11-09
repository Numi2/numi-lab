"""Fail-closed provenance for the frozen native accuracy/cost experiment."""
import hashlib
from pathlib import Path


def _digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _manifest(path):
    values = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid accuracy manifest digest")
        if name in values:
            raise ValueError("duplicate accuracy manifest entry")
        values[name] = digest
    return values


def verify_accuracy_provenance(record, manifest, build):
    """Check both producer identity and shared native benchmark build.

    Two retained execution formats exist: the midpoint record names its frozen
    manifest; the compressed BE record embeds hashes. Neither is evidence for
    a different executable even if a later trajectory happens to look similar.
    """
    manifest, build = Path(manifest), Path(build)
    expected = _manifest(manifest)
    paths = [build / "numi_plane_contact_probe_scaled", build / "NumiMatter.metallib",
             build / "metalrobo_plane_contact_benchmark"]
    for path in paths:
        if expected.get(str(path)) != _digest(path):
            raise ValueError("frozen experiment input changed: " + str(path))
    command = record.get("command", [])
    if len(command) < 2 or command[:2] != [str(paths[0]), str(paths[1])]:
        raise ValueError("accuracy producer is not the frozen native build")
    if "input_sha256" in record:
        observed = record["input_sha256"]
        for path in paths[:2]:
            if observed.get(str(path)) != expected[str(path)]:
                raise ValueError("accuracy producer hash does not match frozen build: " + str(path))
    elif record.get("input_hashes") != manifest.name:
        raise ValueError("accuracy producer has no matching frozen input record")
    return {str(path): expected[str(path)] for path in paths}
