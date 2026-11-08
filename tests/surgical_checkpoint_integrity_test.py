#!/usr/bin/env python3
"""Reject damaged surgical checkpoints before a tissue restore can advance."""
import hashlib
import pathlib
import subprocess
import sys
import tempfile

probe = pathlib.Path(sys.argv[1]).resolve()
source = pathlib.Path(sys.argv[2]).resolve()
original = source.read_bytes()
lines = original.decode("utf-8").splitlines(keepends=True)

def rows(key):
    return [i for i, line in enumerate(lines) if line.startswith(key + "\t")]

twists = rows("thread_twist")
assert len(twists) > 1 and len(rows("matter_snapshot")) == 1
matter = lines[rows("matter_snapshot")[0]].split("\t")[1]
assert pathlib.Path(matter).name == matter

def execute(path):
    return subprocess.run([str(probe), "--tissue-checkpoint-restore-only",
        "--resume-tissue-checkpoint", "tissue-rest", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)

control = execute(source)
assert control.returncode == 0 and "physics_advanced=no" in control.stdout, control.stdout

with tempfile.TemporaryDirectory(prefix="checkpoint-integrity-", dir=source.parent) as tmp:
    root = pathlib.Path(tmp)
    (root / matter).symlink_to(source.parent / matter)
    cases = []
    cases.append(("missing-one-twist", [x for i, x in enumerate(lines) if i != twists[0]],
                  "handoff resume rod twist state is incomplete"))
    cases.append(("missing-all-twists", [x for x in lines if not x.startswith("thread_twist\t")],
                  "handoff resume rod twist state is incomplete"))
    for key, error in [("schema", "duplicate handoff resume schema"),
                       ("phase", "duplicate handoff resume phase"),
                       ("step", "invalid or duplicate resume step"),
                       ("q", "duplicate handoff generalized state"),
                       ("needle_position", "invalid resume needle position"),
                       ("thread_position", "duplicate resume thread node"),
                       ("thread_velocity", "duplicate resume thread node"),
                       ("thread_twist", "duplicate resume thread twist")]:
        idx = rows(key)[0]
        altered = lines.copy()
        altered.insert(idx + 1, lines[idx])
        cases.append(("duplicate-" + key, altered, error))
    downgraded = lines.copy()
    downgraded.insert(rows("schema")[0] + 1, "schema\tnumi.dual-psm-suture-handoff-state.v2\n")
    cases.append(("duplicate-schema-downgrade", downgraded, "duplicate handoff resume schema"))
    # A genuinely old visual-only v2 record can omit every twist. It must
    # pass parsing but still fail the separate v3 Matter-resume requirement.
    legacy = [x.replace("handoff-state.v3", "handoff-state.v2") for x in lines
              if not x.startswith("thread_twist\t")]
    cases.append(("legacy-v2-no-twists", legacy,
                  "surgical tissue resume requires a v3 Matter checkpoint"))
    partial_legacy = legacy + [lines[twists[0]]]
    cases.append(("legacy-v2-partial-twists", partial_legacy,
                  "handoff resume rod twist state is incomplete"))
    # Finite FP64 text can still overflow when narrowed to GPU state floats.
    for key in ("q", "v", "needle_position", "needle_orientation_xyzw",
                "needle_linear_velocity", "needle_angular_velocity",
                "thread_position", "thread_velocity", "thread_twist"):
        idx = rows(key)[0]
        column = 2 if key.startswith("thread_") else 1
        for value, suffix in (("1e100", "positive"), ("-1e100", "negative")):
            altered = lines.copy()
            fields = altered[idx].rstrip("\n").split("\t")
            fields[column] = value
            altered[idx] = "\t".join(fields) + "\n"
            cases.append(("float-overflow-" + key + "-" + suffix, altered,
                          "outside finite float range"))
    for label, content, expected in cases:
        path = root / (label + ".tsv")
        path.write_text("".join(content))
        result = execute(path)
        assert result.returncode != 0 and expected in result.stdout, (label, result.stdout)
        assert "physics_advanced=no" not in result.stdout, (label, result.stdout)
        print(label + ": rejected: " + expected)
assert source.read_bytes() == original
print("original_sha256=" + hashlib.sha256(original).hexdigest())
print("control restored without advancing; original bytes unchanged")
