"""Select the largest certified timestep from a declared native ladder."""
import gzip
import json
from pathlib import Path


def _read_state(prefix):
    plain = Path(str(prefix) + ".json")
    compressed = Path(str(prefix) + ".json.gz")
    if plain.exists() == compressed.exists():
        raise ValueError("missing or ambiguous ladder state: " + str(prefix))
    if compressed.exists():
        with gzip.open(compressed, "rt") as f:
            return json.load(f)
    return json.loads(plain.read_text())


def select_largest_certified(directory, pattern, arm, verify):
    """Only completed, provenance-checked attempts can select a cost arm.

    A rejected or inaccurate attempt remains in the ladder but cannot be
    selected. Incomplete attempts are not treated as tested failures.
    """
    directory = Path(directory)
    endings = ("-execution.json", "-analysis.json", ".json.gz")
    prefixes = set()
    for ending in endings:
        for path in directory.glob(pattern + ending):
            prefixes.add(str(path)[:-len(ending)])
    # Native midpoint traces are uncompressed and may lack a completed record.
    for path in directory.glob(pattern + ".json"):
        name = str(path)
        if not any(name.endswith(s) for s in ("-execution.json", "-analysis.json")):
            prefixes.add(name[:-len(".json")])
    if not prefixes:
        raise ValueError("no declared accuracy attempts for " + arm)
    passing = []
    seen = []
    for name in sorted(prefixes):
        prefix = Path(name)
        execution = Path(name + "-execution.json")
        if not execution.exists():
            raise ValueError("incomplete ladder attempt: " + name)
        record = json.loads(execution.read_text())
        verify(record)
        if (record.get("stopped_reason") is not None or record.get("timed_out") or
                record.get("compression_errors") or record.get("analysis_error")):
            raise ValueError("incomplete ladder attempt: " + name)
        analysis_path = Path(name + "-analysis.json")
        if not analysis_path.exists():
            raise ValueError("missing ladder analysis: " + name)
        analysis = json.loads(analysis_path.read_text())
        command = record.get("command", [])
        if len(command) != 10 or command[5:] != [arm, "0.0002", "physical", "contact", "reference"]:
            raise ValueError("unexpected ladder fixture: " + name)
        h, steps, divisions = float(command[2]), int(command[3]), int(command[4])
        if h <= 0 or steps <= 0 or divisions != 2 or abs(h * steps - .01) > 1e-12:
            raise ValueError("unexpected ladder duration or mesh: " + name)
        passed = record.get("all_gates_pass") is True
        seen.append((prefix.name, h, passed))
        if passed:
            state = _read_state(prefix)
            if (analysis.get("passes") is not True or state.get("arm") != arm or
                    state.get("h") != h or state.get("accepted_steps") != steps or
                    state.get("rejected_steps") != 0):
                raise ValueError("inconsistent passing ladder evidence: " + name)
            passing.append((h, prefix.name))
    if not passing:
        raise ValueError("no certified accuracy attempt for " + arm)
    return max(passing)[1], sorted(seen, key=lambda x: x[1])
