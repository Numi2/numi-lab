#!/usr/bin/env python3
"""Losslessly archive a completed trace without changing the retained original.

Only use after the producer and its execution-record writer have exited.  The
execution record must report an exited producer, including a failed or stopped
run; a changing input or existing output fails closed.  Hashes cover both the exact source bytes and the
archive; a full decompression is checked before the manifest is published.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

root = Path(__file__).resolve().parents[2]
evidence = (root / "docs/evidence/midpoint-first-failure-20261109").resolve()
p = argparse.ArgumentParser()
p.add_argument("prefix", help="completed evidence filename prefix, without .json")
a = p.parse_args()
if not a.prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in a.prefix):
    raise SystemExit("invalid evidence prefix")
source = evidence / (a.prefix + ".json")
execution = evidence / (a.prefix + "-execution.json")
archive = evidence / (a.prefix + ".json.gz")
manifest = evidence / (a.prefix + "-archive.json")
if archive.exists() or manifest.exists():
    raise SystemExit("archive or manifest already exists")
record = json.loads(execution.read_text())
if type(record.get("returncode")) is not int or "stopped_reason" not in record:
    raise SystemExit("execution record does not establish an exited producer")
if not source.is_file() or source.stat().st_size == 0:
    raise SystemExit("completed trace is missing or empty")
if shutil.disk_usage(evidence).free < 3 * 1024**3:
    raise SystemExit("free-space guard")

def identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

before = identity(source.stat())
source_hash = hashlib.sha256()
fd, name = tempfile.mkstemp(prefix=".trace-archive-", dir=evidence)
temporary = Path(name)
try:
    with os.fdopen(fd, "wb") as raw, source.open("rb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                source_hash.update(chunk)
                packed.write(chunk)
        raw.flush()
        os.fsync(raw.fileno())
    if identity(source.stat()) != before:
        raise RuntimeError("source changed during archival")
    restored_hash = hashlib.sha256()
    restored_size = 0
    with gzip.open(temporary, "rb") as restored:
        for chunk in iter(lambda: restored.read(1024 * 1024), b""):
            restored_hash.update(chunk)
            restored_size += len(chunk)
    if restored_hash.digest() != source_hash.digest() or restored_size != before[2]:
        raise RuntimeError("archive restoration check failed")
    if identity(source.stat()) != before:
        raise RuntimeError("source changed before archive publication")
    data = {"schema":"numi.completed-trace-archive.v1", "source":source.name,
            "source_sha256":source_hash.hexdigest(), "source_bytes":before[2],
            "execution_record":execution.name, "execution_sha256":digest_file(execution),
            "producer_returncode":record["returncode"], "producer_stopped_reason":record["stopped_reason"],
            "archive":archive.name, "archive_sha256":digest_file(temporary),
            "archive_bytes":temporary.stat().st_size,
            "decompressed_sha256_verified":True, "original_retained":True}
    # Hard-link publication is exclusive: never overwrite an unrelated file.
    os.link(temporary, archive)
    with manifest.open("x") as out:
        json.dump(data, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps(data, indent=2, sort_keys=True))
finally:
    temporary.unlink(missing_ok=True)
