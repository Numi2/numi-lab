"""Exercise the real pack parser and admission using a copied failure injection."""
import hashlib
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np
from metalrobo.mlx_policy_learning import (
    _LEARNING_PACK_HEADER, _TRANSITION_DTYPE, read_policy_rollout_pack,
)
from metalrobo.native import learning_pack_content_hash
from metalrobo.solver_residual_teacher import distill_residual_frontier

fixture = Path(sys.argv[2])
library = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
original = read_policy_rollout_pack(fixture, library_path=library)
assert not np.any(original.transitions["physics_error"])
with tempfile.TemporaryDirectory() as name:
    root = Path(name)
    copy = root / "injected.rolloutpack"
    raw = bytearray(fixture.read_bytes())
    offset = len(raw) - original.sample_count * _TRANSITION_DTYPE.itemsize
    struct.pack_into("<I", raw, offset + _TRANSITION_DTYPE.fields["physics_error"][1], 1)
    struct.pack_into("<I", raw, offset + _TRANSITION_DTYPE.fields["done"][1], 1)
    copy.write_bytes(raw)
    try:
        read_policy_rollout_pack(copy, library_path=library)
    except ValueError as error:
        assert "fingerprint" in str(error), str(error)
    else:
        raise AssertionError("corrupt pack passed fingerprint")
    magic, version, kind, size, _ = _LEARNING_PACK_HEADER.unpack_from(raw)
    digest = learning_pack_content_hash(
        np.frombuffer(raw, dtype=np.uint8, offset=_LEARNING_PACK_HEADER.size), path=library
    )
    _LEARNING_PACK_HEADER.pack_into(raw, 0, magic, version, kind, size, digest)
    copy.write_bytes(raw)
    injected = read_policy_rollout_pack(copy, library_path=library)
    assert injected.transitions["physics_error"].tolist() == [1] + [0] * (original.sample_count - 1)
    args = dict(
        policy_pack_path=root / "unused.policy", rollout_pack_path=copy,
        evidence_path=root / "missing.evidence", output_policy_path=root / "result.policy",
        output_deployment_policy_path=root / "deploy.policy", output_evidence_path=root / "out.json",
        learning_rate=0.001, update_epochs=1, minibatch_size=1,
        environment_advantage_temperature=1.0, step_advantage_temperature=1.0,
        seed=1, native_library_path=library,
    )
    try:
        distill_residual_frontier(**args)
    except ValueError as error:
        assert "physics failure" in str(error), str(error)
    else:
        raise AssertionError("injected failure accepted")
    args["rollout_pack_path"] = fixture
    try:
        distill_residual_frontier(**args)
    except FileNotFoundError:
        pass  # A clean pack proceeds to the deliberately missing evidence.
    else:
        raise AssertionError("clean control did not reach evidence")
    assert not any((root / n).exists() for n in ("result.policy", "deploy.policy", "out.json"))
    print("source_sha256", sha(fixture))
    print("injected_sha256", sha(copy))
    print("library_sha256", sha(library))
    print("samples", original.sample_count, "environments", original.environment_count)
    print("fingerprint_reject=pass injected_failure_reject=pass clean_control=pass")
